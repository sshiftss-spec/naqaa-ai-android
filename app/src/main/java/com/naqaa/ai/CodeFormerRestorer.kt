package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import com.google.android.gms.tasks.Tasks
import java.io.File
import java.nio.FloatBuffer

/**
 * Experimental CodeFormer ONNX face restoration.
 * Uses a padded square crop from ML Kit face detection; alignment quality varies by pose.
 */
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
        val sourceRect = Rect(left, top, left + side, top + side)
        val destRect = Rect(0, 0, SIZE, SIZE)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        canvas.drawBitmap(source, sourceRect, destRect, paint)

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
                tensors[inputNames[0]] = OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(input), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())
                )
                if (inputNames.size > 1) {
                    tensors[inputNames[1]] = OnnxTensor.createTensor(
                        env, FloatBuffer.wrap(floatArrayOf(fidelity.coerceIn(0f, 1f))), longArrayOf(1)
                    )
                }
                session.run(tensors).use { result ->
                    val raw = result[0].value
                    @Suppress("UNCHECKED_CAST")
                    val output = raw as? Array<Array<Array<FloatArray>>>
                        ?: throw IllegalStateException("صيغة خرج CodeFormer غير مدعومة في هذه النسخة.")
                    if (output[0].size < 3 || output[0][0].size < SIZE || output[0][0][0].size < SIZE) {
                        throw IllegalStateException("أبعاد خرج CodeFormer غير متوافقة.")
                    }
                    for (y in 0 until SIZE) for (x in 0 until SIZE) {
                        val i = y * SIZE + x
                        val r = ((output[0][0][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                        val g = ((output[0][1][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                        val b = ((output[0][2][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                        restoredPixels[i] = Color.rgb(r, g, b)
                    }
                }
            } finally {
                tensors.values.forEach { it.close() }
            }
        }
        val restoredCrop = Bitmap.createBitmap(restoredPixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val output = source.copy(Bitmap.Config.ARGB_8888, true)
        val outputCanvas = Canvas(output)
        val target = Rect(left, top, left + side, top + side)
        outputCanvas.drawBitmap(restoredCrop, Rect(0, 0, SIZE, SIZE), target, Paint(Paint.FILTER_BITMAP_FLAG))
        crop.recycle()
        restoredCrop.recycle()
        return output
    }
}
