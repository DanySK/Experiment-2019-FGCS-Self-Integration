package it.unibo.alchemist.model.implementations.timedistributions


import it.unibo.alchemist.model.implementations.times.DoubleTime
import it.unibo.alchemist.model.interfaces.Environment
import it.unibo.alchemist.model.interfaces.Time
import it.unibo.alchemist.model.interfaces.TimeDistribution
import org.apache.commons.math3.distribution.ExponentialDistribution
import org.apache.commons.math3.random.RandomGenerator

class ExponentialStep<T> @JvmOverloads constructor(
    val randomGenerator: RandomGenerator,
    val baseRate: Double,
    val peakRate: Double,
    val peakStart: Time,
    val peakEnd: Time,
    val startTime: Time = DoubleTime.ZERO_TIME
) : TimeDistribution<T> {

    private var currentTime: Time = startTime
    private val baseExponential = ExponentialDistribution(randomGenerator, 1 / baseRate)
    private val peakExponential = ExponentialDistribution(randomGenerator, 1 / peakRate)
    private var next: Time = currentTime + when(currentTime) {
        in peakStart..peakEnd -> DoubleTime(peakExponential.sample())
        else -> DoubleTime(baseExponential.sample())
    }

    /**
     * Updates the internal status.
     *
     * @param currentTime
     * current time
     * @param executed
     * true if the reaction has just been executed
     * @param param
     * a parameter passed by the reaction
     * @param environment
     * the current environment
     */
    override fun update(currentTime: Time, executed: Boolean, param: Double, environment: Environment<T, *>) {
        this.currentTime = currentTime.takeIf { it > startTime } ?: startTime
        next = currentTime + when(currentTime) {
            in peakStart..peakEnd -> DoubleTime(peakExponential.sample())
            else -> DoubleTime(baseExponential.sample())
        }
    }

    /**
     * @return the next time at which the event will occur
     */
    override fun getNextOccurence() = next

    /**
     * @param currentTime
     * the time at which the cloning operation happened
     * @return an exact copy of this [TimeDistribution]
     */
    override fun clone(currentTime: Time): TimeDistribution<T> =
        ExponentialStep(randomGenerator, baseRate, peakRate, peakStart, peakEnd, currentTime.takeIf { it > startTime } ?: startTime )

    /**
     * @return how many times per time unit the event will happen on average
     */
    override fun getRate() = if (currentTime < peakStart || currentTime > peakEnd) baseRate else peakRate
}