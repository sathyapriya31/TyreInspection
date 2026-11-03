package com.spritle.tyreinspection.ui

import android.content.Context
import android.content.Intent
import android.graphics.BitmapFactory
import android.os.Bundle
import android.text.format.DateFormat
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.spritle.tyreinspection.R
import com.spritle.tyreinspection.data.InspectionRecord
import com.spritle.tyreinspection.data.InspectionRepository
import com.spritle.tyreinspection.databinding.ActivityHistoryBinding
import com.spritle.tyreinspection.databinding.ItemInspectionBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Date

class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding
    private lateinit var repository: InspectionRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)
        repository = InspectionRepository(this)
        binding.toolbar.setNavigationOnClickListener { finish() }
        binding.list.layoutManager = LinearLayoutManager(this)
    }

    override fun onResume() {
        super.onResume()
        lifecycleScope.launch {
            val records = withContext(Dispatchers.IO) { repository.list() }
            binding.list.adapter = Adapter(records)
            binding.empty.visibility = if (records.isEmpty()) View.VISIBLE else View.GONE
        }
    }

    private inner class Adapter(private val items: List<InspectionRecord>) :
        RecyclerView.Adapter<Adapter.Holder>() {

        inner class Holder(val b: ItemInspectionBinding) : RecyclerView.ViewHolder(b.root)

        override fun onCreateViewHolder(parent: ViewGroup, viewType: Int) =
            Holder(ItemInspectionBinding.inflate(LayoutInflater.from(parent.context), parent, false))

        override fun getItemCount() = items.size

        override fun onBindViewHolder(holder: Holder, position: Int) {
            val r = items[position]
            val ctx = holder.itemView.context
            val date = Date(r.timestamp)
            holder.b.title.text = getString(
                R.string.result_header, r.tyrePosition,
                DateFormat.getMediumDateFormat(ctx).format(date) + " " + DateFormat.getTimeFormat(ctx).format(date)
            )
            holder.b.subtitle.text = getString(R.string.history_item_defects, r.result.defects.size)
            holder.b.verdict.text = r.result.verdict.name
            holder.b.verdict.setBackgroundColor(ContextCompat.getColor(ctx, r.result.verdict.colorRes))
            holder.b.thumb.setImageBitmap(
                BitmapFactory.decodeFile(repository.imageFile(r.id).path, BitmapFactory.Options().apply { inSampleSize = 8 })
            )
            holder.itemView.setOnClickListener { startActivity(ResultActivity.open(ctx, r.id)) }
        }
    }

    companion object {
        fun intent(context: Context) = Intent(context, HistoryActivity::class.java)
    }
}
