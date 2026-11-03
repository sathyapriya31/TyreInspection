package com.spritle.tyreinspection.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class VerdictRulesTest {

    private val goodPhoto = ImageQuality(brightness = 120f, sharpness = 300f)
    private val box = Box(0.1f, 0.1f, 0.3f, 0.3f)

    private fun evaluate(
        condition: ClassScore? = null,
        defects: List<Detection> = emptyList(),
        quality: ImageQuality = goodPhoto,
        classifier: Boolean = true,
        detector: Boolean = true,
    ) = VerdictRules.evaluate(condition, defects, quality, classifier, detector)

    @Test
    fun noModels_isUnknown() {
        assertEquals(Verdict.UNKNOWN, evaluate(classifier = false, detector = false).first)
    }

    @Test
    fun goodTyreNoDefects_passes() {
        assertEquals(Verdict.PASS, evaluate(ClassScore("good", 0.95f)).first)
    }

    @Test
    fun criticalDefect_fails() {
        val (verdict, advice) = evaluate(ClassScore("good", 0.9f), listOf(Detection("Bulge", 0.7f, box)))
        assertEquals(Verdict.FAIL, verdict)
        assertTrue(advice.any { "bulge" in it.lowercase() })
    }

    @Test
    fun majorDefect_needsAttention() {
        assertEquals(Verdict.ATTENTION, evaluate(defects = listOf(Detection("worn tread", 0.6f, box))).first)
    }

    @Test
    fun classifierDefective_thresholds() {
        assertEquals(Verdict.FAIL, evaluate(ClassScore("defective", 0.9f)).first)
        assertEquals(Verdict.ATTENTION, evaluate(ClassScore("defective", 0.6f)).first)
    }

    @Test
    fun blurryPhoto_asksForRetake_unlessCriticalDefectSeen() {
        val blurry = ImageQuality(brightness = 120f, sharpness = 10f)
        assertEquals(Verdict.RETAKE, evaluate(ClassScore("good", 0.9f), quality = blurry).first)
        assertEquals(
            Verdict.FAIL,
            evaluate(defects = listOf(Detection("sidewall_damage", 0.8f, box)), quality = blurry).first
        )
    }

    @Test
    fun unknownDefectLabel_isTreatedAsMajor() {
        assertEquals(Severity.MAJOR, VerdictRules.severityOf("weird_spot"))
        assertEquals(Severity.CRITICAL, VerdictRules.severityOf("Sidewall-Damage"))
    }
}
