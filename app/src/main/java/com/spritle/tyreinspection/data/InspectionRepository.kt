package com.spritle.tyreinspection.data

import android.content.Context
import android.graphics.Bitmap
import com.spritle.tyreinspection.ml.Box
import com.spritle.tyreinspection.ml.ClassScore
import com.spritle.tyreinspection.ml.Detection
import com.spritle.tyreinspection.ml.ImageQuality
import com.spritle.tyreinspection.ml.InspectionResult
import com.spritle.tyreinspection.ml.Severity
import com.spritle.tyreinspection.ml.Verdict
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class InspectionRecord(
    val id: String,
    val timestamp: Long,
    /** FL, FR, RL, RR or SPARE. */
    val tyrePosition: String,
    val result: InspectionResult,
)

/** Stores each inspection as `<id>.jpg` + `<id>.json` in app-private storage. */
class InspectionRepository(context: Context) {

    private val dir = File(context.filesDir, "inspections").apply { mkdirs() }

    fun newId(): String = "insp_" + SimpleDateFormat("yyyyMMdd_HHmmss_SSS", Locale.US).format(Date())

    fun imageFile(id: String) = File(dir, "$id.jpg")
    private fun jsonFile(id: String) = File(dir, "$id.json")

    fun saveImage(id: String, bitmap: Bitmap) {
        imageFile(id).outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
    }

    fun save(record: InspectionRecord) {
        jsonFile(record.id).writeText(toJson(record).toString(2))
    }

    fun load(id: String): InspectionRecord? =
        jsonFile(id).takeIf { it.exists() }?.let { runCatching { fromJson(JSONObject(it.readText())) }.getOrNull() }

    fun list(): List<InspectionRecord> =
        dir.listFiles { f -> f.extension == "json" }.orEmpty()
            .mapNotNull { load(it.nameWithoutExtension) }
            .sortedByDescending { it.timestamp }

    fun delete(id: String) {
        imageFile(id).delete()
        jsonFile(id).delete()
    }

    private fun toJson(r: InspectionRecord) = JSONObject().apply {
        put("id", r.id)
        put("timestamp", r.timestamp)
        put("tyrePosition", r.tyrePosition)
        val res = r.result
        put("verdict", res.verdict.name)
        res.condition?.let { put("condition", JSONObject().put("label", it.label).put("confidence", it.confidence.toDouble())) }
        put("defects", JSONArray().apply {
            res.defects.forEach { d ->
                put(JSONObject().apply {
                    put("label", d.label)
                    put("confidence", d.confidence.toDouble())
                    put("severity", d.severity.name)
                    put("box", JSONArray(listOf(d.box.x1, d.box.y1, d.box.x2, d.box.y2).map { it.toDouble() }))
                })
            }
        })
        put("brightness", res.quality.brightness.toDouble())
        put("sharpness", res.quality.sharpness.toDouble())
        put("recommendations", JSONArray(res.recommendations))
        put("inferenceMs", res.inferenceMs)
        put("classifierAvailable", res.classifierAvailable)
        put("detectorAvailable", res.detectorAvailable)
    }

    private fun fromJson(o: JSONObject): InspectionRecord {
        val defects = o.optJSONArray("defects") ?: JSONArray()
        val advice = o.optJSONArray("recommendations") ?: JSONArray()
        return InspectionRecord(
            id = o.getString("id"),
            timestamp = o.getLong("timestamp"),
            tyrePosition = o.optString("tyrePosition"),
            result = InspectionResult(
                verdict = Verdict.valueOf(o.getString("verdict")),
                condition = o.optJSONObject("condition")?.let {
                    ClassScore(it.getString("label"), it.getDouble("confidence").toFloat())
                },
                defects = (0 until defects.length()).map { i ->
                    val d = defects.getJSONObject(i)
                    val b = d.getJSONArray("box")
                    Detection(
                        label = d.getString("label"),
                        confidence = d.getDouble("confidence").toFloat(),
                        box = Box(
                            b.getDouble(0).toFloat(), b.getDouble(1).toFloat(),
                            b.getDouble(2).toFloat(), b.getDouble(3).toFloat()
                        ),
                        severity = Severity.valueOf(d.getString("severity")),
                    )
                },
                quality = ImageQuality(
                    o.getDouble("brightness").toFloat(),
                    o.getDouble("sharpness").toFloat()
                ),
                recommendations = (0 until advice.length()).map { advice.getString(it) },
                inferenceMs = o.optLong("inferenceMs"),
                classifierAvailable = o.optBoolean("classifierAvailable"),
                detectorAvailable = o.optBoolean("detectorAvailable"),
            ),
        )
    }
}
