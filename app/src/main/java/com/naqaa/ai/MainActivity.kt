package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.content.ContentValues
import android.os.Build
import android.provider.MediaStore
import java.io.File
import java.net.URL
import java.io.BufferedInputStream
import java.io.FileOutputStream
import java.nio.FloatBuffer
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoFixHigh
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.ImageSearch
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { NaqaaApp() }
    }
}

@Composable
private fun NaqaaApp() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var original by remember { mutableStateOf<Bitmap?>(null) }
    var enhanced by remember { mutableStateOf<Bitmap?>(null) }
    var busy by remember { mutableStateOf(false) }
    var modelDownloading by remember { mutableStateOf(false) }
    var modelDownloadedBytes by remember { mutableStateOf(0L) }
    var modelTotalBytes by remember { mutableStateOf(340_000_000L) }
    var status by remember { mutableStateOf("اختر صورة للبدء") }
    var strength by remember { mutableFloatStateOf(0.8f) }
    val modelFile = remember { File(context.filesDir, "gfpgan_v14.onnx") }
    var modelReady by remember { mutableStateOf(modelFile.exists() && modelFile.length() > 300_000_000L) }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        if (uri != null) {
            scope.launch {
                busy = true
                status = "جارٍ تحميل الصورة..."
                val bitmap = withContext(Dispatchers.IO) {
                    context.contentResolver.openInputStream(uri)?.use { input ->
                        BitmapFactory.decodeStream(input)
                    }
                }
                original = bitmap
                enhanced = null
                status = if (bitmap != null) "الصورة جاهزة للتحسين" else "تعذر فتح الصورة"
                busy = false
            }
        }
    }

    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = androidx.compose.ui.graphics.Color(0xFF176B5B),
            secondary = androidx.compose.ui.graphics.Color(0xFF5D8D80),
            background = androidx.compose.ui.graphics.Color(0xFFF6F8F7),
            surface = androidx.compose.ui.graphics.Color.White
        )
    ) {
        Scaffold(containerColor = MaterialTheme.colorScheme.background) { padding ->
            Column(
                modifier = Modifier.fillMaxSize().padding(padding)
                    .verticalScroll(rememberScrollState())
                    .padding(horizontal = 20.dp, vertical = 18.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.AutoFixHigh, contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(34.dp))
                    Spacer(Modifier.width(10.dp))
                    Column {
                        Text("نقاء AI", style = MaterialTheme.typography.headlineMedium,
                            fontWeight = FontWeight.Bold)
                        Text("ترميم الوجوه بالذكاء الاصطناعي", style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.secondary)
                    }
                }
                Spacer(Modifier.height(24.dp))

                if (original == null) {
                    Card(
                        modifier = Modifier.fillMaxWidth().height(270.dp),
                        shape = RoundedCornerShape(24.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                    ) {
                        Column(
                            modifier = Modifier.fillMaxSize().padding(24.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.Center
                        ) {
                            Icon(Icons.Default.ImageSearch, contentDescription = null,
                                modifier = Modifier.size(64.dp),
                                tint = MaterialTheme.colorScheme.primary)
                            Spacer(Modifier.height(14.dp))
                            Text("ابدأ بصورة وجه أو شخص", fontWeight = FontWeight.SemiBold,
                                style = MaterialTheme.typography.titleLarge)
                            Spacer(Modifier.height(6.dp))
                            Text("اختر صورة من جهازك. لن يتم رفعها إلى الإنترنت في هذه النسخة.",
                                textAlign = TextAlign.Center,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                            Spacer(Modifier.height(18.dp))
                            Button(onClick = {
                                picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                            }) {
                                Icon(Icons.Default.ImageSearch, contentDescription = null)
                                Spacer(Modifier.width(8.dp))
                                Text("اختيار صورة")
                            }
                        }
                    }
                } else {
                    Text(
                        if (enhanced == null) "الصورة الأصلية" else "مقارنة قبل وبعد التحسين",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.align(Alignment.Start)
                    )
                    Spacer(Modifier.height(10.dp))
                    if (enhanced == null) {
                        Card(shape = RoundedCornerShape(20.dp), modifier = Modifier.fillMaxWidth()) {
                            Image(
                                bitmap = original!!.asImageBitmap(),
                                contentDescription = "الصورة الأصلية",
                                contentScale = ContentScale.Fit,
                                modifier = Modifier.fillMaxWidth().heightIn(max = 420.dp)
                                    .background(androidx.compose.ui.graphics.Color(0xFFE8ECEA))
                            )
                        }
                        Text("الأصل — اضغط تحسين لعرض المقارنة",
                            color = MaterialTheme.colorScheme.secondary,
                            modifier = Modifier.padding(top = 8.dp))
                    } else {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(10.dp)
                        ) {
                            Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("قبل", fontWeight = FontWeight.SemiBold)
                                Spacer(Modifier.height(6.dp))
                                Card(shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
                                    Image(
                                        bitmap = original!!.asImageBitmap(),
                                        contentDescription = "الصورة قبل التحسين",
                                        contentScale = ContentScale.Fit,
                                        modifier = Modifier.fillMaxWidth().height(230.dp)
                                            .background(androidx.compose.ui.graphics.Color(0xFFE8ECEA))
                                    )
                                }
                            }
                            Column(modifier = Modifier.weight(1f), horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("بعد", fontWeight = FontWeight.SemiBold,
                                    color = MaterialTheme.colorScheme.primary)
                                Spacer(Modifier.height(6.dp))
                                Card(shape = RoundedCornerShape(14.dp), modifier = Modifier.fillMaxWidth()) {
                                    Image(
                                        bitmap = enhanced!!.asImageBitmap(),
                                        contentDescription = "الصورة بعد التحسين المحلي التجريبي",
                                        contentScale = ContentScale.Fit,
                                        modifier = Modifier.fillMaxWidth().height(230.dp)
                                            .background(androidx.compose.ui.graphics.Color(0xFFE8ECEA))
                                    )
                                }
                            }
                        }
                        Text("ترميم GFPGAN يعيد بناء تفاصيل الوجه تقديريًا. للحصول على أفضل نتيجة استخدم صورة وجه قريبة ومتمركزة.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.secondary,
                            modifier = Modifier.padding(top = 8.dp))
                    }
                    Spacer(Modifier.height(18.dp))
                    Text("قوة دمج الترميم: ${(strength * 100).toInt()}%",
                        modifier = Modifier.align(Alignment.Start))
                    Slider(value = strength, onValueChange = { strength = it })
                    Button(
                        modifier = Modifier.fillMaxWidth().height(52.dp),
                        enabled = !busy,
                        onClick = {
                            val source = original ?: return@Button
                            scope.launch {
                                busy = true
                                try {
                                    if (!modelReady) {
                                        modelDownloading = true
                                        modelDownloadedBytes = 0L
                                        modelTotalBytes = 340_000_000L
                                        status = "جارٍ الاتصال بخادم النموذج..."
                                        try {
                                            withContext(Dispatchers.IO) {
                                                downloadGfpganModel(modelFile) { downloaded, total ->
                                                    scope.launch(Dispatchers.Main.immediate) {
                                                        modelDownloadedBytes = downloaded
                                                        if (total > 0L) modelTotalBytes = total
                                                        status = "جارٍ تنزيل نموذج GFPGAN من الإنترنت..."
                                                    }
                                                }
                                            }
                                        } finally {
                                            modelDownloading = false
                                        }
                                        modelReady = true
                                        status = "اكتمل تنزيل النموذج. أصبح جاهزًا للعمل دون إنترنت."
                                    } else {
                                        status = "جارٍ ترميم الوجه على الهاتف..."
                                        enhanced = withContext(Dispatchers.Default) {
                                            GfpganRestorer.restore(source, modelFile, strength)
                                        }
                                        status = "اكتمل ترميم الوجه محليًا دون رفع الصورة"
                                    }
                                } catch (e: Exception) {
                                    status = "تعذّر إكمال العملية: ${e.message ?: "خطأ غير معروف"}"
                                } finally {
                                    busy = false
                                }
                            }
                        }
                    ) {
                        Icon(Icons.Default.AutoFixHigh, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(if (modelReady) "ترميم الوجه بالذكاء الاصطناعي" else "تنزيل نموذج الترميم (340 MB)")
                    }
                    if (modelDownloading) {
                        Spacer(Modifier.height(12.dp))
                        Card(
                            modifier = Modifier.fillMaxWidth(),
                            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                        ) {
                            Column(Modifier.fillMaxWidth().padding(16.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(20.dp),
                                        strokeWidth = 2.dp
                                    )
                                    Spacer(Modifier.width(10.dp))
                                    Text("تنزيل نموذج الذكاء الاصطناعي", fontWeight = FontWeight.SemiBold)
                                }
                                Spacer(Modifier.height(12.dp))
                                val progress = (modelDownloadedBytes.toFloat() /
                                    modelTotalBytes.coerceAtLeast(1L).toFloat()).coerceIn(0f, 1f)
                                LinearProgressIndicator(
                                    progress = { progress },
                                    modifier = Modifier.fillMaxWidth()
                                )
                                Spacer(Modifier.height(8.dp))
                                Text(
                                    "${"%.1f".format(java.util.Locale.US, modelDownloadedBytes / 1_000_000.0)} MB من " +
                                        "${"%.1f".format(java.util.Locale.US, modelTotalBytes / 1_000_000.0)} MB  •  ${(progress * 100).toInt()}%",
                                    style = MaterialTheme.typography.bodyMedium,
                                    fontWeight = FontWeight.Medium
                                )
                                Text(
                                    "يتحدّث العداد مع وصول البيانات فعليًا من الإنترنت. أبقِ التطبيق مفتوحًا حتى يكتمل التنزيل.",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(
                        modifier = Modifier.fillMaxWidth(),
                        enabled = enhanced != null && !busy,
                        onClick = {
                            val bitmapToSave = enhanced ?: return@OutlinedButton
                            scope.launch {
                                busy = true
                                status = "جارٍ حفظ الصورة..."
                                val saved = withContext(Dispatchers.IO) {
                                    saveBitmapToGallery(context, bitmapToSave)
                                }
                                status = if (saved) "تم حفظ الصورة في المعرض" else "تعذر حفظ الصورة"
                                busy = false
                            }
                        }
                    ) {
                        Icon(Icons.Default.SaveAlt, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("حفظ الصورة المحسّنة")
                    }
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(
                        modifier = Modifier.fillMaxWidth(),
                        onClick = {
                            picker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                        }
                    ) { Text("اختيار صورة أخرى") }
                    TextButton(onClick = {
                        original = null
                        enhanced = null
                        status = "اختر صورة للبدء"
                    }) { Text("مسح الصورة") }
                }

                Spacer(Modifier.height(20.dp))
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
                ) {
                    Column(Modifier.padding(16.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.Cloud, contentDescription = null,
                                tint = MaterialTheme.colorScheme.primary)
                            Spacer(Modifier.width(8.dp))
                            Text("التحسين السحابي", fontWeight = FontWeight.SemiBold)
                        }
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "سيُضاف بعد إعداد خادم آمن ونموذج متخصص للوجوه. غير مفعّل في هذه النسخة، ولا تُرسل الصور لأي خدمة.",
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            style = MaterialTheme.typography.bodyMedium
                        )
                    }
                }
                Spacer(Modifier.height(14.dp))
                if (busy) LinearProgressIndicator(Modifier.fillMaxWidth())
                Text(status, modifier = Modifier.padding(top = 8.dp),
                    color = MaterialTheme.colorScheme.secondary,
                    textAlign = TextAlign.Center)
                Spacer(Modifier.height(24.dp))
                Text("نسخة تجريبية • ترميم الوجه دون إنترنت بعد تنزيل النموذج", style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

/**
 * Simple on-device unsharp mask prototype.
 * This is not a trained face-restoration model and cannot recover lost detail.
 */
private fun saveBitmapToGallery(context: android.content.Context, bitmap: Bitmap): Boolean {
    val filename = "Naqaa_${System.currentTimeMillis()}.jpg"
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, filename)
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/NaqaaAI")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }
    }
    val resolver = context.contentResolver
    val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return false
    return try {
        resolver.openOutputStream(uri)?.use { output ->
            if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output)) {
                throw IllegalStateException("JPEG encoding failed")
            }
        } ?: throw IllegalStateException("Could not open output stream")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val done = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
            resolver.update(uri, done, null, null)
        }
        true
    } catch (_: Exception) {
        resolver.delete(uri, null, null)
        false
    }
}

private const val GFP_GAN_URL =
    "https://huggingface.co/ykk648/face_lib/resolve/main/face_restore/gfpgan/GFPGANv1.4.onnx"

private fun downloadGfpganModel(destination: File, onProgress: (Long, Long) -> Unit) {
    val temp = File(destination.parentFile, destination.name + ".download")
    val connection = URL(GFP_GAN_URL).openConnection().apply {
        connectTimeout = 20_000
        readTimeout = 60_000
        setRequestProperty("User-Agent", "NaqaaAI-Android")
    }
    try {
        connection.getInputStream().use { rawInput ->
            BufferedInputStream(rawInput).use { input ->
                FileOutputStream(temp).use { output ->
                    val declaredLength = connection.contentLengthLong
                    val totalBytes = if (declaredLength > 0L) declaredLength else 340_000_000L
                    val buffer = ByteArray(1024 * 1024)
                    var downloaded = 0L
                    var lastReported = 0L
                    while (true) {
                        val count = input.read(buffer)
                        if (count < 0) break
                        output.write(buffer, 0, count)
                        downloaded += count
                        if (downloaded - lastReported >= 1024 * 1024 || downloaded == totalBytes) {
                            onProgress(downloaded, totalBytes)
                            lastReported = downloaded
                        }
                    }
                    onProgress(downloaded, totalBytes)
                }
            }
        }
        if (temp.length() < 300_000_000L) {
            throw IllegalStateException("ملف النموذج غير مكتمل؛ أعد التنزيل")
        }
        if (destination.exists()) destination.delete()
        if (!temp.renameTo(destination)) throw IllegalStateException("تعذر حفظ ملف النموذج")
    } catch (e: Exception) {
        temp.delete()
        throw e
    }
}

/**
 * GFPGAN v1.4 ONNX inference. The model expects a 512x512 face-like crop
 * normalized to [-1,1]. This MVP uses a centered square crop, so a close-up,
 * centered face gives the best result; it is not a full face-detection/alignment pipeline.
 */
private object GfpganRestorer {
    fun restore(src: Bitmap, model: File, blend: Float): Bitmap {
        val size = minOf(src.width, src.height)
        require(size >= 64) { "الصورة صغيرة جدًا للترميم" }
        val left = (src.width - size) / 2
        val top = (src.height - size) / 2
        val crop = Bitmap.createBitmap(src, left, top, size, size)
        val inputBitmap = Bitmap.createScaledBitmap(crop, 512, 512, true)
        val pixels = IntArray(512 * 512)
        inputBitmap.getPixels(pixels, 0, 512, 0, 0, 512, 512)
        val floats = FloatArray(3 * 512 * 512)
        for (y in 0 until 512) {
            for (x in 0 until 512) {
                val p = pixels[y * 512 + x]
                val i = y * 512 + x
                floats[i] = (Color.red(p) / 127.5f) - 1f
                floats[512 * 512 + i] = (Color.green(p) / 127.5f) - 1f
                floats[2 * 512 * 512 + i] = (Color.blue(p) / 127.5f) - 1f
            }
        }

        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(2)
        env.createSession(model.absolutePath, options).use { session ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), longArrayOf(1, 3, 512, 512)).use { tensor ->
                session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val output = result[0].value as Array<Array<Array<FloatArray>>>
                    val restoredPixels = IntArray(512 * 512)
                    for (y in 0 until 512) {
                        for (x in 0 until 512) {
                            val i = y * 512 + x
                            val r = ((output[0][0][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            val g = ((output[0][1][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            val b = ((output[0][2][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            restoredPixels[i] = Color.rgb(r, g, b)
                        }
                    }
                    val restored512 = Bitmap.createBitmap(restoredPixels, 512, 512, Bitmap.Config.ARGB_8888)
                    val restoredCrop = Bitmap.createScaledBitmap(restored512, size, size, true)
                    val outputBitmap = src.copy(Bitmap.Config.ARGB_8888, true)
                    val originalPixels = IntArray(size * size)
                    val restored = IntArray(size * size)
                    crop.getPixels(originalPixels, 0, size, 0, 0, size, size)
                    restoredCrop.getPixels(restored, 0, size, 0, 0, size, size)
                    val amount = blend.coerceIn(0f, 1f)
                    for (i in originalPixels.indices) {
                        val a = originalPixels[i]
                        val b = restored[i]
                        fun mix(c1: Int, c2: Int) = (c1 + (c2 - c1) * amount).toInt().coerceIn(0, 255)
                        originalPixels[i] = Color.rgb(
                            mix(Color.red(a), Color.red(b)),
                            mix(Color.green(a), Color.green(b)),
                            mix(Color.blue(a), Color.blue(b))
                        )
                    }
                    outputBitmap.setPixels(originalPixels, 0, size, left, top, size, size)
                    return outputBitmap
                }
            }
        }
    }
}
