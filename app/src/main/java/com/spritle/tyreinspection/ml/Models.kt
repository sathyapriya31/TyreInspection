package com.spritle.tyreinspection.ml

/** Axis-aligned box in normalized image coordinates (0..1). */
data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
    val width get() = x2 - x1
    val height get() = y2 - y1
    val area get() = width * height
}

enum class Severity { CRITICAL, MAJOR, MINOR }

/** A single defect found by the detector model. */
data class Detection(
    val label: String,
    val confidence: Float,
    val box: Box,
    val severity: Severity = VerdictRules.severityOf(label),
)

/** One class probability from the condition classifier. */
data class ClassScore(val label: String, val confidence: Float)

enum class Verdict {
    /** No defects and the classifier considers the tyre good. */
    PASS,
    /** Minor/major issues: drive with care and have the tyre checked soon. */
    ATTENTION,
    /** Critical damage: the tyre should not be used. */
    FAIL,
    /** Photo is too dark or blurry to judge. */
    RETAKE,
    /** No model is installed, so nothing could be judged. */
    UNKNOWN,
}

/**
 * Photo quality metrics.
 * @param brightness mean luma, 0..255
 * @param sharpness variance of the Laplacian on a downscaled grey image; higher is sharper
 */
data class ImageQuality(val brightness: Float, val sharpness: Float) {
    val tooDark get() = brightness < MIN_BRIGHTNESS
    val tooBright get() = brightness > MAX_BRIGHTNESS
    val blurry get() = sharpness < MIN_SHARPNESS
    val acceptable get() = !tooDark && !tooBright && !blurry

    companion object {
        const val MIN_BRIGHTNESS = 45f
        const val MAX_BRIGHTNESS = 225f
        const val MIN_SHARPNESS = 60f
    }
}

data class InspectionResult(
    val verdict: Verdict,
    /** Top class from the condition classifier, or null when that model is absent. */
    val condition: ClassScore?,
    val defects: List<Detection>,
    val quality: ImageQuality,
    val recommendations: List<String>,
    val inferenceMs: Long,
    val classifierAvailable: Boolean,
    val detectorAvailable: Boolean,
)
