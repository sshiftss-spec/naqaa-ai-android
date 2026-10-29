package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import com.google.android.gms.tasks.Tasks
import java.io.File
import java.nio.DoubleBuffer
import java.nio.FloatBuffer

/** Experimental CodeFormer ONNX face restoration with tensor types read from model metadata. */
internal object CodeFormerRestorer {
    private const val SIZE = 512

    fun restore(source: Bitmap, modelFile: File, fidelity: Float = 0.7f): Bitmap {
        val detector = com.google.mlkit.vision.face.FaceDetection.getClient(
            com.google.mlkit.vision.face.FaceDetectorOptions.Builder()
                .setPerformanceMode(com.google.mlkit.vision.face.FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .build()
        )
        val faces = try {
            Tasks.await(detector.process(com.google.mlkit.vision.common.InputImage.fromBitmap(source, 0)))
        } finally {
            detector.close()
        }
        val face = faces.maxByOrNull { it.boundingBox.width().toLong() * it.boundingBox.height().toLong() }
            ?: throw IllegalArgumentException("لم يُعثر على وجه واضح لترميمه باستخدام CodeFormer.")

        val box = face.boundingBox
        val side = (maxOf(box.width(), box.height()) * 1.8f).toInt().coerceAtLeast(32)
        val cx = (box.left + box.right) / 2
        val cy = (box.top + box.bottom) / 2
        val left = cx - side / 2
        val top = cy - side / 2
        val crop = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(crop)
        canvas.drawColor(Color.rgb(127, 127, 127))
        canvas.drawBitmap(source, Rect(left, top, left + side, top + side),
            Rect(0, 0, SIZE, SIZE), Paint(Paint.FILTER_BITMAP_FLAG))

        val pixels = IntArray(SIZE * SIZE)
        crop.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val plane = SIZE * SIZE
        val input = FloatArray(3 * plane)
        for (i in pixels.indices) {
            val p = pixels[i]
            input[i] = Color.red(p) / 127.5f - 1f
            input[plane + i] = Color.green(p) / 127.5f - 1f
            input[2 * plane + i] = Color.blue(p) / 127.5f - 1f
        }

        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply { setIntraOpNumThreads(2) }
        val restoredPixels = IntArray(plane)
        env.createSession(modelFile.absolutePath, options).use { session ->
            val inputNames = session.inputNames.toList()
            if (inputNames.isEmpty()) throw IllegalStateException("نموذج CodeFormer لا يحتوي على مدخلات.")
            val tensors = mutableMapOf<String, OnnxTensor>()
            try {
                for ((index, name) in inputNames.withIndex()) {
                    val info = session.inputInfo[name]?.info as? TensorInfo
                        ?: throw IllegalStateException("تعذر قراءة وصف المدخل: $name")
                    val shape = if (index == 0) longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
                        else info.shape.map { if (it <= 0L) 1L else it }.toLongArray()
                    val values = if (index == 0) input else floatArrayOf(fidelity.coerceIn(0f, 1f))
                    tensors[name] = when (info.type.toString().lowercase()) {
                        "double" -> OnnxTensor.createTensor(env,
                            DoubleBuffer.wrap(DoubleArray(values.size) { values[it].toDouble() }), shape)
                        "float" -> OnnxTensor.createTensor(env, FloatBuffer.wrap(values), shape)
                        else -> throw IllegalStateException("نوع مدخل CodeFormer غير مدعوم: ${info.type} ($name)")
                    }
                }
                session.run(tensors).use { result ->
                    val raw = result[0].value
                    val info = result[0].info as? TensorInfo
                        ?: throw IllegalStateException("تعذر قراءة وصف خرج CodeFormer.")
                    val output = when (info.type.toString().lowercase()) {
                        "double" -> {
                            @Suppress("UNCHECKED_CAST")
                            val a = raw as? Array<Array<Array<DoubleArray>>>
                                ?: throw IllegalStateException("صيغة خرج CodeFormer غير متوقعة.")
                            Array(3) { c -> FloatArray(plane) { i -> a[0][c][i / SIZE][i % SIZE].toFloat() } }
                        }
                        "float" -> {
                            @Suppress("UNCHECKED_CAST")
                            val a = raw as? Array<Array<Array<FloatArray>>>
                                ?: throw IllegalStateException("صيغة خرج CodeFormer غير متوقعة.")
                            Array(3) { c -> FloatArray(plane) { i -> a[0][c][i / SIZE][i % SIZE] } }
                        }
                        else -> throw IllegalStateException("نوع خرج CodeFormer غير مدعوم: ${info.type}")
                    }
                    for (i in 0 until plane) {
                        val r = ((output[0][i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                        val g = ((output[1][i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                        val b = ((output[2][i] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                        restoredPixels[i] = Color.rgb(r, g, b)
                    }
                }
            } finally {
                tensors.values.forEach { it.close() }
            }
        }
        val restoredCrop = Bitmap.createBitmap(restoredPixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        Canvas(output).drawBitmap(restoredCrop, Rect(0, 0, SIZE, SIZE),
            Rect(left, top, left + side, top + side), Paint(Paint.FILTER_BITMAP_FLAG))
        crop.recycle()
        restoredCrop.recycle()
        return output
    }
}
