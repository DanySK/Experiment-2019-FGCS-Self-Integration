import com.uchuhimo.konf.Config
import com.uchuhimo.konf.ConfigSpec
import com.uchuhimo.konf.source.yaml
import org.protelis.lang.datatype.DatatypeFactory
import org.protelis.lang.datatype.Tuple
import org.protelis.vm.ExecutionContext
import java.lang.IllegalStateException
import java.util.WeakHashMap

//@file:JvmName("Tasks")
typealias Instructions = Long
typealias MIPS = Long
typealias SchedulingPolicy = (List<Allocation>, Task) -> Unit

private val taskmanagers: MutableMap<ExecutionContext, Scheduler> = WeakHashMap()
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
val ExecutionContext.scheduler: Scheduler get() = taskmanagers.computeIfAbsent(this) {
    val type = executionEnvironment["cpuType"] ?: throw IllegalStateException("No cpuType specified for node $deviceUID")
    Scheduler(cpus[type] ?: throw IllegalStateException("No cpu in $cpus available for $type"), AllocateOnMin)
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
    private var completedTasks: List<CompletedTask> = emptyList()
    @JvmStatic
    fun completedTasks(): Tuple = completedTasks.asTuple // Tuple<CompletedTask>
    @JvmStatic
    fun freeMips(context: ExecutionContext): MIPS = context.scheduler.freeMIPS
    @JvmStatic
    fun enqueue(context: ExecutionContext, tasks: Iterable<Task>): Unit = tasks.forEach(context.scheduler::enqueue)
    @JvmStatic
    fun update(context: ExecutionContext, maximumTaskAge: Double): Unit {
        val now = context.currentTime.toDouble()
        completedTasks =
            completedTasks.filter { it.completionTime + maximumTaskAge < now } +
            context.scheduler.update(context.deltaTime.toDouble())
               .map { CompletedTask(it, now) }
    }
    @JvmStatic
    fun waitingTasks(context: ExecutionContext): Tuple = context.scheduler.allocatedTasks.asSequence().flatMap {
            it.tasks.asSequence().map { it.first }
        }.toList().asTuple // Tuple<Task>
    @JvmStatic
    fun runningTasks(context: ExecutionContext): Tuple = context.scheduler.allocatedTasks.asSequence()
        .flatMap { allocation -> allocation.tasks.asSequence().map { it.first } }
        .toList().asTuple // Tuple<Task>
}

data class Task(val id: Long, val instructions: Instructions) {
    companion object {
        private var idGenerator = 0L
        @JvmStatic
        fun task(instructions: Long) = Task(idGenerator++, instructions)
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
fun MIPS.forSeconds(seconds: Double): Instructions = (seconds * 1_000_000 * this).toLong()
class Allocation(cpu: CPU) {
    private val mips: MIPS = cpu.mips / cpu.cores
    var tasks: List<Pair<Task, Instructions>> = emptyList()
        private set
    val enqueuedInstructions: Instructions
        get() = tasks.map { it.second }.sum()
    fun allocate(task: Task) {
        tasks += task to task.instructions
    }
    fun update(deltaTime: Double): List<Task> {
        var instructionsLeft = mips.forSeconds(deltaTime)
        val taskMap = tasks.groupBy { (_, weight) ->
            instructionsLeft -= weight
            instructionsLeft >= 0
        }
        val newqueue = taskMap[false] ?: emptyList()
        val partlyExecuted = newqueue.getOrNull(0)
        val first = partlyExecuted
            ?.copy(second = partlyExecuted.second + instructionsLeft)
            ?.let { listOf(it) }
            ?: emptyList()
        tasks = first + newqueue.drop(1)
        return taskMap[true]?.map { it.first } ?: emptyList()
    }
}
class Scheduler(val cpu: CPU, private val policy: SchedulingPolicy) {
    val allocatedTasks: List<Allocation> = generateSequence { Allocation(cpu) }
        .take(cpu.cores).toList()
    val freeMIPS: MIPS get() = allocatedTasks.let { load ->
        val freeCores = load.count { it.tasks.isEmpty() }
        if (freeCores > 0) {
            freeCores * cpu.mips / cpu.cores
        } else {
            -load.map { it.enqueuedInstructions }.sum()
        }
    }
    fun enqueue(task: Task) {
        policy(allocatedTasks, task)
    }
    fun update(deltaTime: Double): List<Task> = allocatedTasks.flatMap { it.update(deltaTime) }
}
