package com.spritle.tyreinspection.ml

/**
 * Turns raw model outputs into a safety verdict and human-readable advice.
 *
 * Pure Kotlin (no Android types) so the rules are unit-tested on the JVM.
 * Label names are matched case-insensitively with spaces/hyphens treated as underscores,
 * so "Sidewall Damage", "sidewall-damage" and "sidewall_damage" are the same defect.
 */
object VerdictRules {

    /** Classifier confidence for a "defective" label at which the tyre fails outright. */
    const val DEFECTIVE_FAIL_CONFIDENCE = 0.85f
    /** Classifier confidence for a "defective" label at which the tyre needs attention. */
    const val DEFECTIVE_ATTENTION_CONFIDENCE = 0.5f

    private val SEVERITY = mapOf(
        "bulge" to Severity.CRITICAL,
        "cut" to Severity.CRITICAL,
        "sidewall_damage" to Severity.CRITICAL,
        "puncture" to Severity.CRITICAL,
        "exposed_cord" to Severity.CRITICAL,
        "crack" to Severity.MAJOR,
        "worn_tread" to Severity.MAJOR,
        "foreign_object" to Severity.MAJOR,
        "uneven_wear" to Severity.MINOR,
    )

    private val ADVICE = mapOf(
        "bulge" to "Sidewall bulge means the internal cords are broken. Replace the tyre before driving further.",
        "cut" to "Deep cut found. Have a technician check for exposed cords; replace the tyre if any are visible.",
        "sidewall_damage" to "Sidewall damage cannot be repaired safely. Replace the tyre.",
        "puncture" to "Puncture found. Tread-area punctures up to 6 mm can be plugged and patched; otherwise replace.",
        "exposed_cord" to "Steel or fabric cords are exposed. Replace the tyre immediately.",
        "crack" to "Cracking / dry rot found. Replace if cracks are deeper than about 2 mm or the tyre is over 6 years old.",
        "worn_tread" to "Tread looks worn. Measure with a depth gauge: the legal minimum is usually 1.6 mm and replacement is advised below 3 mm.",
        "foreign_object" to "Object embedded in the tyre. Remove it and check for air leaks (soapy water test).",
        "uneven_wear" to "Uneven wear suggests wrong pressure or misalignment. Check pressure and get a wheel alignment.",
    )

    private val DEFECTIVE_CLASS_LABELS = setOf("defective", "damaged", "bad", "worn", "unsafe")

    fun normalize(label: String) = label.trim().lowercase().replace(' ', '_').replace('-', '_')

    /** Unknown defect labels are treated as MAJOR, which errs on the side of safety. */
    fun severityOf(label: String): Severity = SEVERITY[normalize(label)] ?: Severity.MAJOR

    fun isDefectiveClass(label: String) = normalize(label) in DEFECTIVE_CLASS_LABELS

    fun evaluate(
        condition: ClassScore?,
        defects: List<Detection>,
        quality: ImageQuality,
        classifierAvailable: Boolean,
        detectorAvailable: Boolean,
    ): Pair<Verdict, List<String>> {
        if (!classifierAvailable && !detectorAvailable) {
            return Verdict.UNKNOWN to listOf(
                "No AI model is installed. Add tyre_condition.tflite and/or tyre_defects.tflite to the app assets (see ml/README.md)."
            )
        }

        val advice = mutableListOf<String>()
        val hasCritical = defects.any { it.severity == Severity.CRITICAL }

        // A blurry/dark photo can hide damage, so only a clear critical finding overrides a retake.
        if (!quality.acceptable && !hasCritical) {
            if (quality.tooDark) advice += "Photo is too dark. Turn on the torch or move to better light."
            if (quality.tooBright) advice += "Photo is overexposed. Avoid direct sunlight or glare on the tyre."
            if (quality.blurry) advice += "Photo is blurry. Hold the phone steady about 30–50 cm from the tyre and tap to focus."
            return Verdict.RETAKE to advice
        }

        defects.map { normalize(it.label) }.distinct().forEach { label ->
            advice += ADVICE[label] ?: "Possible defect \"$label\" found. Have the tyre inspected by a technician."
        }

        val defectiveConfidence = condition
            ?.takeIf { isDefectiveClass(it.label) }
            ?.confidence ?: 0f

        val verdict = when {
            hasCritical -> Verdict.FAIL
            defectiveConfidence >= DEFECTIVE_FAIL_CONFIDENCE -> Verdict.FAIL
            defects.any { it.severity == Severity.MAJOR } -> Verdict.ATTENTION
            defectiveConfidence >= DEFECTIVE_ATTENTION_CONFIDENCE -> Verdict.ATTENTION
            defects.isNotEmpty() -> Verdict.ATTENTION
            else -> Verdict.PASS
        }

        if (defectiveConfidence >= DEFECTIVE_ATTENTION_CONFIDENCE && defects.isEmpty()) {
            advice += "The AI rates this tyre as defective (${(defectiveConfidence * 100).toInt()}%) but could not localise the damage. " +
                "Look closely for cracks, bulges or cuts, or get it checked by a technician."
        }
        if (verdict == Verdict.PASS) {
            advice += "No defects found. Keep checking pressure monthly and tread depth with a gauge."
        }
        if (!detectorAvailable) {
            advice += "Defect detector model not installed: only the overall condition was checked."
        }
        return verdict to advice
    }
}
