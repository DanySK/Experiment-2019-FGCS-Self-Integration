import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import java.io.ByteArrayOutputStream

/*
 * DEFAULT GRADLE BUILD FOR ALCHEMIST SIMULATOR
 */

plugins {
    kotlin("jvm") version "1.3.60"
    application
}

repositories {
    mavenCentral()
    /* 
     * The following repositories contain beta features and should be added for experimental mode only
     * 
     * maven("https://dl.bintray.com/alchemist-simulator/Alchemist/")
     * maven("https://dl.bintray.com/protelis/Protelis/")
     */
}
/*
 * Only required if you plan to use Protelis, remove otherwise
 */
sourceSets {
    main {
        resources {
            srcDir("src/main/protelis")
        }
    }
}
dependencies {
    // it is highly recommended to replace the '+' symbol with a fixed version
    implementation("it.unibo.alchemist:alchemist:9.3.0")
    implementation("org.jetbrains.kotlin:kotlin-stdlib:1.3.60")
    implementation("com.uchuhimo:konf-core:+")
    implementation("com.uchuhimo:konf-yaml:+")
}
tasks.withType<KotlinCompile> {
    kotlinOptions {
        jvmTarget = "1.8"
    }
}

val alchemistGroup = "Run Alchemist"
/*
 * This task is used to run all experiments in sequence
 */
val showAll by tasks.register<DefaultTask>("showAll") {
    group = alchemistGroup
    description = "Launches all simulations in graphic mode (unless the CI environment variable is set to \"true\")"
}
val runAllExperiments by tasks.register<DefaultTask>("runAllExperiments") {
    group = alchemistGroup
    description = "Launches all simulations in batch mode, reproducing the experiment"
}

// Heap size estimation for batches
val maxHeap: Long? by project
val heap: Long = maxHeap ?:
    if (System.getProperty("os.name").toLowerCase().contains("linux")) {
        ByteArrayOutputStream().use { output ->
            exec {
                executable = "bash"
                args = listOf("-c", "cat /proc/meminfo | grep MemAvailable | grep -o '[0-9]*'")
                standardOutput = output
            }
            output.toString().trim().toLong() / 1024
        }
        .also { println("Detected ${it}MB RAM available.") }  * 9 / 10
    } else {
        // Guess 16GB RAM of which 2 used by the OS
        14 * 1024L
    }
val taskSizeFromProject: Int? by project
val taskSize = taskSizeFromProject ?: 512
val threadCount = maxOf(1, minOf(Runtime.getRuntime().availableProcessors(), heap.toInt() / taskSize ))

/*
 * Scan the folder with the simulation files, and create a task for each one of them.
 */
File(rootProject.rootDir.path + "/src/main/yaml").listFiles()
    .filter { it.extension == "yml" }
    .sortedBy { it.nameWithoutExtension }
    .forEach {
        fun basetask(name: String, additionalConfiguration: JavaExec.() -> Unit = {}) = tasks.register<JavaExec>(name) {
            group = alchemistGroup
            description = "Launches simulation ${it.nameWithoutExtension}"
            main = "it.unibo.alchemist.Alchemist"
            classpath = sourceSets["main"].runtimeClasspath
            args(
                "-y", it.absolutePath,
                "-g", "effects/${it.nameWithoutExtension}.aes"
            )
            if (System.getenv("CI") == "true") {
                args("-hl", "-t", "10")
            }
            this.additionalConfiguration()
        }
        val capitalizedName = it.nameWithoutExtension.capitalize()
        val basetask by basetask("show$capitalizedName")
        val batchTask by basetask("run$capitalizedName") {
            description = "Launches batch experiments for $capitalizedName"
            jvmArgs("-XX:+AggressiveHeap")
            maxHeapSize = "${heap}m"
            File("data").mkdirs()
            val variables = listOf("seed", "meanTaskSize", "smoothing", "grain", "peakFrequency").toTypedArray()
            args(
                "-e", "data/${it.nameWithoutExtension}",
                "-b",
                "-var", *variables,
                "-p", threadCount
            )
        }
        runAllExperiments.dependsOn(batchTask)
        showAll.dependsOn(basetask)
    }

