import com.uchuhimo.konf.Config
import com.uchuhimo.konf.ConfigSpec
import com.uchuhimo.konf.emptyMutableMap
import com.uchuhimo.konf.source.yaml
import it.unibo.alchemist.model.interfaces.Environment
import it.unibo.alchemist.protelis.AlchemistExecutionContext
import javafx.application.Application.launch
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.protelis.lang.datatype.DatatypeFactory
import org.protelis.lang.datatype.Field
import org.protelis.lang.datatype.Tuple
import org.protelis.vm.ExecutionContext
import java.lang.IllegalStateException
import java.util.Collections
import java.util.WeakHashMap
import kotlin.math.min

//@file:JvmName("Tasks")
typealias Instructions = Long
typealias MIPS = Long
typealias SchedulingPolicy = (List<Allocation>, Task) -> Unit

private val taskmanagers: MutableMap<Environment<*, *>, MutableMap<ExecutionContext, Scheduler>> = WeakHashMap()
val cpus: Map<String, CPU> = Config {
        addSpec(CPU)
        addSpec(ProcessorList)
    }.from.yaml
    .inputStream(Thread.currentThread().contextClassLoader.getResourceAsStream("cpus.yaml"))
    .let { it[ProcessorList.cpus] }
    .groupBy { it.type }
    .mapValues { (type, cpulist) -> when {
        cpulist.size == 1 -> cpulist.first()
        else -> throw IllegalStateException("Multiple CPUs specified for type $type: $cpulist")
    } }
private val mutex = Mutex()
val ExecutionContext.scheduler: Scheduler get() = when (this) {
        is AlchemistExecutionContext<*> -> runBlocking {
            mutex.withLock {
                taskmanagers
                    .computeIfAbsent(getEnvironmentAccess()) { mutableMapOf() }
                    .computeIfAbsent(this@scheduler) {
                        val type = executionEnvironment["cpuType"]
                            ?: throw IllegalStateException("No cpuType specified for node $deviceUID")
                        Scheduler(
                            cpus[type] ?: throw IllegalStateException("No cpu in $cpus available for $type"),
                            AllocateOnMin
                        )
                    }
            }
        }
        else -> throw IllegalStateException()
    }

object ProcessorList : ConfigSpec("") {
    val cpus by required<List<CPU>>()
}

fun main() {
    val config = Config {
        addSpec(CPU)
        addSpec(ProcessorList)
    }
    val cpus = config.from.yaml
        .inputStream(Thread.currentThread().contextClassLoader.getResourceAsStream("cpus.yaml"))
    val allCPUs = cpus[ProcessorList.cpus]
    println(allCPUs)
}

val List<Any>.asTuple
    get() = DatatypeFactory.createTuple(this)

data class CompletedTask(val task: Task, val completionTime: Double)
object TaskManager {
    @JvmStatic
    fun completedTasks(context: ExecutionContext): Tuple = context.scheduler.completedTasks.asTuple // Tuple<CompletedTask>
    @JvmStatic
    fun cleanupCompletedTasks(context: ExecutionContext, tasks: Iterable<Task>): Unit = context.scheduler.signalTaskCompletion(tasks)
    @JvmStatic
    fun freeMips(context: ExecutionContext): MIPS = context.scheduler.freeMIPS
    @JvmStatic
    fun enqueue(context: ExecutionContext, tasks: Iterable<Task>): Unit = tasks.forEach(context.scheduler::enqueue)
    @JvmStatic
    fun update(context: ExecutionContext, maximumTaskAge: Double): Unit {
        context.scheduler.update(context.currentTime.toDouble(), context.deltaTime.toDouble(), maximumTaskAge)
    }
    @JvmStatic
    fun waitingTasks(context: ExecutionContext): Tuple = context.scheduler.localTasks.asTuple // Tuple<Task>
    @JvmStatic
    fun runningTasks(context: ExecutionContext): Tuple = allocationInfo(context).asSequence()
        .flatMap { allocation -> allocation.tasks.asSequence().map { it.first } }
        .toList().asTuple // Tuple<Task>
    @JvmStatic
    fun allocationInfo(context: ExecutionContext) = context.scheduler.allocatedTasks
}

data class Task(val id: Long, val instructions: Instructions) {
    companion object {
        private var idGenerator = 0L

        @JvmStatic
        fun newTask(context: ExecutionContext, instructions: Long, maxTasks: Int) = Task(idGenerator++, instructions).also {
            fun ExecutionContext.environment() = (this as AlchemistExecutionContext<*>).getEnvironmentAccess()
            val scheduler = taskmanagers.getOrElse(context.environment(), Collections::emptyMap)
                .asSequence()
                .find { (ctx, _) -> ctx.deviceUID == context.deviceUID }
                ?.value
            if (scheduler != null) {
                if (scheduler.localTasks.size < maxTasks) {
                    scheduler.localTasks += it
                } else {
                    with (context.executionEnvironment) {
                        val previous = if (has("drop")) get("drop") as Number else 1
                        context.executionEnvironment.put("drop", previous.toInt() + 1 )
                    }
                }
            }
        }
    }
}
data class CPU(val model: String, val cores: Int, val mips: Long, val type: String) {
    companion object : ConfigSpec() {
        val cores by required<Int>()
        val mips by required<Int>()
        val model by required<String>()
        val type by required<String>()
    }
}
object AllocateOnMin: SchedulingPolicy {
    override fun invoke(allocatedTasks: List<Allocation>, task: Task) = allocatedTasks
        .minBy { it.enqueuedInstructions }
        ?.allocate(task)
        ?: throw IllegalStateException("unable to compute min on: $allocatedTasks")
}
fun MIPS.forSeconds(seconds: Double): Instructions = (seconds * this).toLong()
/**
 * Allocation of a sequence of tasks on a logical core of a given CPU
 */
class Allocation(cpu: CPU) {
    val mips: MIPS = cpu.mips / cpu.cores
    var tasks: List<Pair<Task, Instructions>> = emptyList()
        private set
    val enqueuedInstructions: Instructions
        get() = tasks.map { it.second }.sum()
    fun allocate(task: Task) {
        tasks += task to task.instructions
    }
    /**
     * Runs the allocated tasks for [deltaTime], returning the list of completed tasks
     */
    fun update(deltaTime: Double): List<Task> {
        var instructionsLeft = mips.forSeconds(deltaTime)
        val completed = tasks.takeWhile { (_, weight) ->
            instructionsLeft -= weight
            instructionsLeft >= 0
        }
        val incomplete = tasks.drop(completed.size + 1)
        val possiblyPartlyExecuted = tasks.getOrNull(completed.size)
        val first = possiblyPartlyExecuted
            ?.copy(second = -instructionsLeft)
            ?.let { listOf(it) }
            ?: emptyList()
        tasks = first + incomplete
//        require(tasks.all { (task, i) -> task.instructions >= i })
        return completed.map { it.first }
    }
    override fun toString() = tasks.map { (task, instructions) -> "${task.id}:$instructions" }.toString()
}

/**
 * CPU scheduling: allocates tasks on the [cpu] cores according to a [policy]
 */
class Scheduler(val cpu: CPU, private val policy: SchedulingPolicy) {
    /**
     * The list of [Allocation]s, one per core.
     */
    val allocatedTasks: List<Allocation> = generateSequence { Allocation(cpu) }
        .take(cpu.cores).toList()
    var completedTasks: List<CompletedTask> = emptyList()
        private set
    var localTasks: List<Task> = emptyList()
    val allTasks get() = allocatedTasks.asSequence()
        .flatMap { it.tasks.asSequence() }
        .map { it.first }
        .toSet()
    private var newTasks: List<CompletedTask> = emptyList()
    val freeMIPS: MIPS get() = allocatedTasks.let { load ->
//        val freeCores = load.count { it.tasks.isEmpty() }
//        val base = 2 * freeCores * cpu.mips / cpu.cores
//        base - load.map { min(it.enqueuedInstructions, it.mips) }.sum()
        load.count { it.tasks.isEmpty() } * cpu.mips / cpu.cores
    }
    fun enqueue(task: Task) {
        if (!allTasks.contains(task)) {
            policy(allocatedTasks, task)
        }
    }
    private fun updateCores(deltaTime: Double): List<Task> = allocatedTasks.flatMap { it.update(deltaTime) }
    /**
     * Lets the core run for [deltaTime], adds completed tasks to the completion list, filters out old tasks.
     */
    fun update(now: Double, deltaTime: Double, maximumTaskAge: Double): Unit {
        completedTasks =
            // remove tasks older than a threshold (they have been completed long ago)
            completedTasks.filter { it.completionTime + maximumTaskAge > now } +
            // let the cores run, adding completed tasks to the list
            updateCores(deltaTime).map { CompletedTask(it, now) }
    }
    /**
     * tells this scheduler that the provided tasks (local or not) have been completed
     */
    fun signalTaskCompletion(tasks: Iterable<Task>) {
        localTasks -= tasks
    }
}

object Assertions {
    @JvmStatic
    fun metricIsConsistent(metric: Field<Double>) = require(
        metric.localValue == 0.0 && metric.values().all { it in 0.0..1.1 }
    ) {
        """invalid field $metric: local value expected 0, is: ${metric.localValue} (${metric.localValue == 0.0}).
|          Other values in ]0, 1.1[: ${metric.toMap().mapValues { (id, v) -> "$v -> ${v > 0 && v < 1.1 }" }}""".trimMargin()
    }
    @JvmStatic
    fun nonNegative(number: Number) = require(number.toDouble() >= 0)
}