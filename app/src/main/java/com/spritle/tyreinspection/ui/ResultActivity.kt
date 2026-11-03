package com.spritle.tyreinspection.ui

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.text.format.DateFormat
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.lifecycle.lifecycleScope
import com.spritle.tyreinspection.R
import com.spritle.tyreinspection.data.InspectionRecord
import com.spritle.tyreinspection.data.InspectionRepository
import com.spritle.tyreinspection.databinding.ActivityResultBinding
import com.spritle.tyreinspection.ml.TyreInspector
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Date

/**
 * Inspection report. For a new photo it runs the full inspection and saves the record;
 * for a history entry it just loads the saved record.
 */
class ResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityResultBinding
    private lateinit var repository: InspectionRepository
    private var record: InspectionRecord? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityResultBinding.inflate(layoutInflater)
        setContentView(binding.root)
        repository = InspectionRepository(this)

        val id = intent.getStringExtra(EXTRA_ID) ?: return finish()
        val position = intent.getStringExtra(EXTRA_POSITION).orEmpty()

        binding.doneButton.setOnClickListener { finish() }
        binding.deleteButton.setOnClickListener {
            repository.delete(id)
            finish()
        }
        binding.shareButton.setOnClickListener { record?.let(::share) }

        lifecycleScope.launch {
            val photo = withContext(Dispatchers.IO) { BitmapFactory.decodeFile(repository.imageFile(id).path) }
            if (photo == null) return@launch finish()
            binding.photo.setImageBitmap(photo)

            // Saved records are reused (rotation, history); only brand-new photos are analysed.
            val loaded = repository.load(id) ?: withContext(Dispatchers.Default) {
                val result = TyreInspector(this@ResultActivity).use { it.inspect(photo) }
                InspectionRecord(id, System.currentTimeMillis(), position, result).also(repository::save)
            }
            record = loaded
            binding.overlay.setResults(loaded.result.defects, photo.width, photo.height)
            render(loaded)
        }
    }

    private fun render(r: InspectionRecord) {
        val res = r.result
        binding.progress.visibility = View.GONE
        binding.verdict.setText(res.verdict.labelRes)
        binding.verdict.setBackgroundColor(ContextCompat.getColor(this, res.verdict.colorRes))

        binding.header.text = getString(
            R.string.result_header, r.tyrePosition,
            DateFormat.getMediumDateFormat(this).format(Date(r.timestamp)) + " " +
                DateFormat.getTimeFormat(this).format(Date(r.timestamp))
        )
        binding.condition.text = res.condition?.let {
            getString(R.string.condition_value, it.label, (it.confidence * 100).toInt())
        } ?: getString(R.string.condition_unavailable)
        binding.defects.text = if (res.defects.isEmpty()) getString(R.string.no_defects)
        else res.defects.joinToString("\n") {
            "• " + getString(
                R.string.defect_value, it.label.replace('_', ' '), it.severity.name.lowercase(),
                (it.confidence * 100).toInt()
            )
        }
        binding.advice.text = res.recommendations.joinToString("\n") { "• $it" }
        binding.quality.text = getString(
            R.string.quality_value, res.quality.brightness, res.quality.sharpness, res.inferenceMs.toInt()
        )

        binding.details.visibility = View.VISIBLE
        binding.shareButton.isEnabled = true
        binding.deleteButton.isEnabled = true
    }

    private fun share(r: InspectionRecord) {
        val text = buildString {
            appendLine(getString(R.string.share_subject))
            appendLine(getString(r.result.verdict.labelRes))
            appendLine("${getString(R.string.tyre_position)}: ${r.tyrePosition}")
            appendLine("${getString(R.string.section_condition)}: ${binding.condition.text}")
            appendLine("${getString(R.string.section_defects)}:\n${binding.defects.text}")
            appendLine("${getString(R.string.section_advice)}:\n${binding.advice.text}")
            append(getString(R.string.disclaimer))
        }
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", repository.imageFile(r.id))
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "image/jpeg"
            putExtra(Intent.EXTRA_SUBJECT, getString(R.string.share_subject))
            putExtra(Intent.EXTRA_TEXT, text)
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        startActivity(Intent.createChooser(send, getString(R.string.share)))
    }

    companion object {
        private const val EXTRA_ID = "id"
        private const val EXTRA_POSITION = "position"

        fun newInspection(context: Context, id: String, position: String) =
            Intent(context, ResultActivity::class.java)
                .putExtra(EXTRA_ID, id)
                .putExtra(EXTRA_POSITION, position)

        fun open(context: Context, id: String) =
            Intent(context, ResultActivity::class.java).putExtra(EXTRA_ID, id)
    }
}
