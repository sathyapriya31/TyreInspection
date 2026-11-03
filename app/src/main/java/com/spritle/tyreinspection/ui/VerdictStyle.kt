package com.spritle.tyreinspection.ui

import androidx.annotation.ColorRes
import androidx.annotation.StringRes
import com.spritle.tyreinspection.R
import com.spritle.tyreinspection.ml.Verdict

@get:ColorRes
val Verdict.colorRes: Int
    get() = when (this) {
        Verdict.PASS -> R.color.verdict_pass
        Verdict.ATTENTION -> R.color.verdict_attention
        Verdict.FAIL -> R.color.verdict_fail
        Verdict.RETAKE -> R.color.verdict_retake
        Verdict.UNKNOWN -> R.color.verdict_unknown
    }

@get:StringRes
val Verdict.labelRes: Int
    get() = when (this) {
        Verdict.PASS -> R.string.verdict_pass
        Verdict.ATTENTION -> R.string.verdict_attention
        Verdict.FAIL -> R.string.verdict_fail
        Verdict.RETAKE -> R.string.verdict_retake
        Verdict.UNKNOWN -> R.string.verdict_unknown
    }
