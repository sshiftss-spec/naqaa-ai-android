package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import kotlin.math.max
import android.net.Uri
import android.content.Intent
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
import kotlinx.coroutines.delay
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
    var status by remember { mutableStateOf("اختر صورة للبدء") }
    var downloadProgress by remember { mutableFloatStateOf(0f) }
    var downloadBytes by remember { mutableLongStateOf(0L) }
    var downloadTotal by remember { mutableLongStateOf(-1L) }
    var modelMode by remember { mutableStateOf("both") }
    var compareMode by remember { mutableStateOf(true) }
    val modelFile = remember { File(context.filesDir, "nafnet_deblur.onnx") }
    val codeFormerFile = remember { File(context.filesDir, "codeformer.onnx") }
    var modelReady by remember { mutableStateOf(modelFile.exists() && modelFile.length() > 80_000_000L) }
    var codeFormerReady by remember { mutableStateOf(codeFormerFile.exists() && codeFormerFile.length() > 300_000_000L) }

    // Observe background model downloads and refresh the progress bar when the app returns.
    LaunchedEffect(Unit) {
        val prefs = context.getSharedPreferences(ModelDownloadService.PREFS, 0)
        while (true) {
            val nafState = prefs.getString("nafnet_state", "idle") ?: "idle"
            val faceState = prefs.getString("codeformer_state", "idle") ?: "idle"
            val faceActive = faceState == "downloading"
            val prefix = if (faceActive) "codeformer_" else "nafnet_"
            downloadBytes = prefs.getLong(prefix + "downloaded", 0L)
            downloadTotal = prefs.getLong(prefix + "total", -1L)
            if (downloadTotal > 0L) downloadProgress = (downloadBytes.toFloat() / downloadTotal.toFloat()).coerceIn(0f, 1f)
            if (modelFile.exists() && modelFile.length() > 80_000_000L) modelReady = true
            if (codeFormerFile.exists() && codeFormerFile.length() > 300_000_000L) codeFormerReady = true
            if (faceActive) {
                status = "جارٍ تنزيل CodeFormer في الخلفية..."
            } else if (nafState == "downloading" && !modelReady) {
                status = "جارٍ تنزيل NAFNet في الخلفية..."
            } else if (faceState == "error" && !codeFormerReady) {
                status = "تعذّر تنزيل CodeFormer: " + (prefs.getString("codeformer_error", "خطأ غير معروف") ?: "خطأ غير معروف")
            } else if (nafState == "error" && !modelReady) {
                status = "تعذّر تنزيل NAFNet: " + (prefs.getString("nafnet_error", "خطأ غير معروف") ?: "خطأ غير معروف")
            }
            kotlinx.coroutines.delay(750)
        }
    }

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
                        Text("إزالة الضبابية من الصورة كاملة", style = MaterialTheme.typography.bodyMedium,
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
                            Text("اختر صورة لإزالة الضبابية", fontWeight = FontWeight.SemiBold,
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
                    } else if (!compareMode) {
                        Card(shape = RoundedCornerShape(20.dp), modifier = Modifier.fillMaxWidth()) {
                            Image(
                                bitmap = enhanced!!.asImageBitmap(),
                                contentDescription = "النتيجة المحسنة",
                                contentScale = ContentScale.Fit,
                                modifier = Modifier.fillMaxWidth().heightIn(max = 420.dp)
                                    .background(androidx.compose.ui.graphics.Color(0xFFE8ECEA))
                            )
                        }
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
                        Text("يكتشف التطبيق أكبر وجه في الصورة ويحاول تصحيح ميله قبل الترميم. قد تتغير بعض الملامح لأن نموذج الترميم يعيد بناء التفاصيل تقديريًا.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.secondary,
                            modifier = Modifier.padding(top = 8.dp))
                    }
                    Spacer(Modifier.height(18.dp))
                    Text(
                        "اختر طريقة التحسين",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.align(Alignment.Start)
                    )
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Column(Modifier.padding(12.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                RadioButton(selected = modelMode == "both", onClick = { modelMode = "both" })
                                Column {
                                    Text("النموذجان معًا (افتراضي)", fontWeight = FontWeight.SemiBold)
                                    Text("NAFNet للصورة كاملة ثم CodeFormer للوجه", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                RadioButton(selected = modelMode == "nafnet", onClick = { modelMode = "nafnet" })
                                Text("NAFNet فقط — الصورة كاملة")
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                RadioButton(selected = modelMode == "codeformer", onClick = { modelMode = "codeformer" })
                                Text("CodeFormer فقط — ترميم الوجه")
                            }
                        }
                    }
                    Spacer(Modifier.height(12.dp))
                    Text(
                        "NAFNet يعالج الصورة كاملة. CodeFormer يركّز على الوجه وقد يعيد بناء بعض التفاصيل تقديريًا.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.secondary
                    )
                    Button(
                        modifier = Modifier.fillMaxWidth().height(52.dp),
                        enabled = !busy,
                        onClick = {
                            val source = original ?: return@Button
                            scope.launch {
                                busy = true
                                try {
                                    val prefs = context.getSharedPreferences(ModelDownloadService.PREFS, 0)
                                    downloadProgress = 0f
                                    suspend fun ensureDownloaded(isFaceModel: Boolean) {
                                        val file = if (isFaceModel) codeFormerFile else modelFile
                                        val ready = if (isFaceModel) codeFormerReady else modelReady
                                        val minimum = if (isFaceModel) 300_000_000L else 80_000_000L
                                        val prefix = if (isFaceModel) "codeformer_" else "nafnet_"
                                        val action = if (isFaceModel) ModelDownloadService.ACTION_START_CODEFORMER else ModelDownloadService.ACTION_START
                                        val label = if (isFaceModel) "CodeFormer" else "NAFNet"
                                        if (ready || (file.exists() && file.length() > minimum)) return
                                        val state = prefs.getString(prefix + "state", "idle")
                                        if (state != "downloading") {
                                            status = "بدء تنزيل $label في الخلفية..."
                                            val intent = Intent(context, ModelDownloadService::class.java).setAction(action)
                                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) context.startForegroundService(intent)
                                            else context.startService(intent)
                                        }
                                        while (!(file.exists() && file.length() > minimum)) {
                                            kotlinx.coroutines.delay(500)
                                            if (prefs.getString(prefix + "state", "idle") == "error") {
                                                throw IllegalStateException(prefs.getString(prefix + "error", "تعذّر تنزيل $label"))
                                            }
                                        }
                                        if (isFaceModel) codeFormerReady = true else modelReady = true
                                    }
                                    if (modelMode == "both" || modelMode == "nafnet") ensureDownloaded(false)
                                    if (modelMode == "both" || modelMode == "codeformer") ensureDownloaded(true)
                                    status = "جارٍ تطبيق النماذج المختارة..."
                                    enhanced = withContext(Dispatchers.Default) {
                                        var result = source
                                        if (modelMode == "both" || modelMode == "nafnet") {
                                            result = NafNetDeblurRestorer.restore(result, modelFile)
                                        }
                                        if (modelMode == "both" || modelMode == "codeformer") {
                                            result = CodeFormerRestorer.restore(result, codeFormerFile)
                                        }
                                        result
                                    }
                                    compareMode = true
                                    status = "اكتملت المعالجة محليًا. قارن النتيجة بالأصل."
                                } catch (e: Exception) {
                                    status = "تعذرت إزالة الضبابية: ${e.message ?: "خطأ غير معروف"}"
                                } finally {
                                    busy = false
                                }
                            }
                        }
                    ) {
                        Icon(Icons.Default.AutoFixHigh, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text(if ((modelMode == "nafnet" && modelReady) || (modelMode == "codeformer" && codeFormerReady) || (modelMode == "both" && modelReady && codeFormerReady)) "ابدأ التحسين" else "تنزيل النماذج المطلوبة بالخلفية")
                    }
                    Spacer(Modifier.height(8.dp))
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(
                        modifier = Modifier.fillMaxWidth(),
                        enabled = enhanced != null && !busy,
                        onClick = {
                            compareMode = !compareMode
                        }
                    ) {
                        Text(if (compareMode) "عرض النتيجة فقط" else "مقارنة الأصل والنتيجة")
                    }
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
                val downloadingAnyModel = status.contains("تنزيل") && original != null
                if (downloadingAnyModel) {
                    LinearProgressIndicator(
                        progress = { downloadProgress },
                        modifier = Modifier.fillMaxWidth()
                    )
                    val progressLabel = if (downloadTotal > 0L) {
                        val downloadedMb = downloadBytes / (1024L * 1024L)
                        val totalMb = downloadTotal / (1024L * 1024L)
                        "تنزيل النموذج: " + downloadedMb + " / " + totalMb + " MB — " + (downloadProgress * 100).toInt() + "%"
                    } else "جارٍ تنزيل النموذج في الخلفية..."
                    Text(progressLabel, modifier = Modifier.padding(top = 6.dp), textAlign = TextAlign.Center)
                } else if (busy) {
                    LinearProgressIndicator(Modifier.fillMaxWidth())
                }
                Text(status, modifier = Modifier.padding(top = 8.dp),
                    color = MaterialTheme.colorScheme.secondary,
                    textAlign = TextAlign.Center)
                Spacer(Modifier.height(24.dp))
                Text("نسخة تجريبية • إزالة ضبابية الصورة كاملةً على الجهاز", style = MaterialTheme.typography.labelSmall,
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

private const val RESTOREFORMER_PP_URL =
    "https://huggingface.co/Saimon8420/restoreformer-pp-web/resolve/main/restoreformer_pp_fp16w.onnx"

/**
 * Experimental RestoreFormer++ inference using the model’s RGB [-1, 1] ONNX interface.
 * Face crop/placement remains approximate in this preview build; quality must be tested on-device.
 * Falls back with a clear error if no face is detected instead of restoring the image center.
 */
private object RestoreFormerRestorer {
    private const val SIZE = 512

    /**
     * Whole-frame local deblurring approximation using an unsharp mask and mild contrast.
     * This runs on every pixel before face restoration, so body, clothing, hair, and background
     * are also affected. It can reduce mild blur but cannot perfectly invert a strong Gaussian blur
     * or recover details that were lost; a general-purpose trained deblurring model would be needed.
     * Rows are processed incrementally to avoid allocating several full-resolution pixel arrays.
     */
    private fun enhanceWholeImage(src: Bitmap, strength: Float): Bitmap {
        val w = src.width
        val h = src.height
        val result = src.copy(Bitmap.Config.ARGB_8888, true)
        val previous = IntArray(w)
        val current = IntArray(w)
        val next = IntArray(w)
        val output = IntArray(w)

        for (y in 0 until h) {
            val y0 = (y - 1).coerceAtLeast(0)
            val y1 = (y + 1).coerceAtMost(h - 1)
            src.getPixels(previous, 0, w, 0, y0, w, 1)
            src.getPixels(current, 0, w, 0, y, w, 1)
            src.getPixels(next, 0, w, 0, y1, w, 1)
            for (x in 0 until w) {
                val p = current[x]
                val left = current[(x - 1).coerceAtLeast(0)]
                val right = current[(x + 1).coerceAtMost(w - 1)]
                val up = previous[x]
                val down = next[x]

                fun improve(channel: (Int) -> Int): Int {
                    val center = channel(p).toFloat()
                    val neighborhood = (
                        channel(left) + channel(right) + channel(up) + channel(down)
                    ) / 4f
                    val sharpened = center + strength.coerceIn(0f, 1f) * (center - neighborhood)
                    val contrasted = (sharpened - 127.5f) * 1.025f + 127.5f
                    return contrasted.toInt().coerceIn(0, 255)
                }

                output[x] = Color.argb(
                    Color.alpha(p),
                    improve(Color::red),
                    improve(Color::green),
                    improve(Color::blue)
                )
            }
            result.setPixels(output, 0, w, 0, y, w, 1)
        }
        return result
    }

    // FFHQ five-point template used by facexlib / RestoreFormer pipelines.
    private val template = arrayOf(
        floatArrayOf(192.98138f, 239.94708f), // image-left eye
        floatArrayOf(318.90277f, 240.1936f),  // image-right eye
        floatArrayOf(256.63416f, 314.01935f), // nose
        floatArrayOf(201.26117f, 371.41043f), // image-left mouth corner
        floatArrayOf(313.08905f, 371.15118f)  // image-right mouth corner
    )

    /**
     * Least-squares affine fit from five detected source landmarks to the FFHQ template.
     * Returns Android Matrix values mapping source-image coordinates into 512x512 model space.
     */
    private fun fitAffine(source: List<Pair<Float, Float>>, target: Array<FloatArray>): android.graphics.Matrix {
        require(source.size == 5 && target.size == 5) { "Five facial landmarks are required." }
        val normal = Array(3) { DoubleArray(3) }
        val rhsX = DoubleArray(3)
        val rhsY = DoubleArray(3)
        for (i in 0 until 5) {
            val x = source[i].first.toDouble()
            val y = source[i].second.toDouble()
            val row = doubleArrayOf(x, y, 1.0)
            val tx = target[i][0].toDouble()
            val ty = target[i][1].toDouble()
            for (r in 0..2) {
                rhsX[r] += row[r] * tx
                rhsY[r] += row[r] * ty
                for (c in 0..2) normal[r][c] += row[r] * row[c]
            }
        }

        fun solve(matrix: Array<DoubleArray>, rhs: DoubleArray): DoubleArray {
            val a = Array(3) { r -> DoubleArray(4) { c -> if (c < 3) matrix[r][c] else rhs[r] } }
            for (col in 0..2) {
                var pivot = col
                for (r in col + 1..2) if (kotlin.math.abs(a[r][col]) > kotlin.math.abs(a[pivot][col])) pivot = r
                if (kotlin.math.abs(a[pivot][col]) < 1e-9) {
                    throw IllegalArgumentException("تعذرت محاذاة الوجه؛ جرّب صورة أوضح.")
                }
                val tmp = a[col]
                a[col] = a[pivot]
                a[pivot] = tmp
                val divisor = a[col][col]
                for (c in col..3) a[col][c] /= divisor
                for (r in 0..2) {
                    if (r == col) continue
                    val factor = a[r][col]
                    for (c in col..3) a[r][c] -= factor * a[col][c]
                }
            }
            return DoubleArray(3) { a[it][3] }
        }

        val sx = solve(normal, rhsX)
        val sy = solve(normal, rhsY)
        val values = floatArrayOf(
            sx[0].toFloat(), sx[1].toFloat(), sx[2].toFloat(),
            sy[0].toFloat(), sy[1].toFloat(), sy[2].toFloat(),
            0f, 0f, 1f
        )
        return android.graphics.Matrix().apply { setValues(values) }
    }

    fun restore(src: Bitmap, model: File, blend: Float, wholeImageSharpness: Float): Bitmap {
        val detectorOptions = com.google.mlkit.vision.face.FaceDetectorOptions.Builder()
            .setPerformanceMode(com.google.mlkit.vision.face.FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(com.google.mlkit.vision.face.FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(com.google.mlkit.vision.face.FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .build()
        val detector = com.google.mlkit.vision.face.FaceDetection.getClient(detectorOptions)
        val faces = try {
            com.google.android.gms.tasks.Tasks.await(
                detector.process(com.google.mlkit.vision.common.InputImage.fromBitmap(src, 0))
            )
        } finally {
            detector.close()
        }
        val face = faces.maxByOrNull { it.boundingBox.width().toLong() * it.boundingBox.height().toLong() }
            ?: throw IllegalArgumentException("لم يُعثر على وجه. جرّب صورة يظهر فيها الوجه بوضوح.")

        fun landmark(type: Int): Pair<Float, Float> {
            val point = face.getLandmark(type)?.position
                ?: throw IllegalArgumentException("لم تُكتشف نقاط الوجه الخمس بوضوح. جرّب صورة أمامية بإضاءة جيدة.")
            return Pair(point.x, point.y)
        }

        // Sort by image x so the template order is consistent regardless of detector naming convention.
        val eyes = listOf(
            landmark(com.google.mlkit.vision.face.FaceLandmark.LEFT_EYE),
            landmark(com.google.mlkit.vision.face.FaceLandmark.RIGHT_EYE)
        ).sortedBy { it.first }
        val mouth = listOf(
            landmark(com.google.mlkit.vision.face.FaceLandmark.MOUTH_LEFT),
            landmark(com.google.mlkit.vision.face.FaceLandmark.MOUTH_RIGHT)
        ).sortedBy { it.first }
        val points = listOf(
            eyes[0],
            eyes[1],
            landmark(com.google.mlkit.vision.face.FaceLandmark.NOSE_BASE),
            mouth[0],
            mouth[1]
        )

        val forward = fitAffine(points, template)
        val inverse = android.graphics.Matrix()
        if (!forward.invert(inverse)) {
            throw IllegalArgumentException("تعذرت محاذاة الوجه؛ جرّب صورة أخرى.")
        }

        // Warp the original image into the model's expected FFHQ five-point geometry.
        val aligned = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
        val alignedCanvas = android.graphics.Canvas(aligned)
        alignedCanvas.drawColor(Color.rgb(127, 127, 127))
        alignedCanvas.drawBitmap(src, forward, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))

        val pixels = IntArray(SIZE * SIZE)
        aligned.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val floats = FloatArray(3 * SIZE * SIZE)
        for (y in 0 until SIZE) {
            for (x in 0 until SIZE) {
                val p = pixels[y * SIZE + x]
                val i = y * SIZE + x
                floats[i] = (Color.red(p) / 127.5f) - 1f
                floats[SIZE * SIZE + i] = (Color.green(p) / 127.5f) - 1f
                floats[2 * SIZE * SIZE + i] = (Color.blue(p) / 127.5f) - 1f
            }
        }

        val restoredPixels = IntArray(SIZE * SIZE)
        val env = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions()
        options.setIntraOpNumThreads(2)
        env.createSession(model.absolutePath, options).use { session ->
            OnnxTensor.createTensor(env, FloatBuffer.wrap(floats), longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { tensor ->
                session.run(mapOf(session.inputNames.first() to tensor)).use { result ->
                    @Suppress("UNCHECKED_CAST")
                    val output = result[0].value as Array<Array<Array<FloatArray>>>
                    for (y in 0 until SIZE) {
                        for (x in 0 until SIZE) {
                            val i = y * SIZE + x
                            val r = ((output[0][0][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            val g = ((output[0][1][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            val b = ((output[0][2][y][x] + 1f) * 127.5f).toInt().coerceIn(0, 255)
                            restoredPixels[i] = Color.rgb(r, g, b)
                        }
                    }
                }
            }
        }

        val restoredAligned = Bitmap.createBitmap(restoredPixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)

        // Inverse-warp the restored face back to the exact original-image coordinate system.
        val overlay = Bitmap.createBitmap(src.width, src.height, Bitmap.Config.ARGB_8888)
        val overlayCanvas = android.graphics.Canvas(overlay)
        overlayCanvas.drawBitmap(
            restoredAligned,
            inverse,
            android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
        )

        // Keep the whole frame gently enhanced; the AI restoration is blended over the face region.
        val outputBitmap = enhanceWholeImage(src, wholeImageSharpness)
        val box = face.boundingBox
        val left = (box.left - box.width() * 0.35f).toInt().coerceAtLeast(0)
        val top = (box.top - box.height() * 0.35f).toInt().coerceAtLeast(0)
        val right = (box.right + box.width() * 0.35f).toInt().coerceAtMost(src.width)
        val bottom = (box.bottom + box.height() * 0.45f).toInt().coerceAtMost(src.height)
        val width = right - left
        val height = bottom - top
        if (width <= 0 || height <= 0) return outputBitmap

        val originalRegion = IntArray(width * height)
        val overlayRegion = IntArray(width * height)
        outputBitmap.getPixels(originalRegion, 0, width, left, top, width, height)
        overlay.getPixels(overlayRegion, 0, width, left, top, width, height)
        val amount = blend.coerceIn(0f, 1f)
        val cx = (box.left + box.right) / 2f
        val cy = (box.top + box.bottom) / 2f
        val rx = (box.width() * 0.78f).coerceAtLeast(1f)
        val ry = (box.height() * 0.90f).coerceAtLeast(1f)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val i = y * width + x
                val p = overlayRegion[i]
                if (Color.alpha(p) == 0) continue
                val px = left + x + 0.5f
                val py = top + y + 0.5f
                val dx = (px - cx) / rx
                val dy = (py - cy) / ry
                val distance = kotlin.math.sqrt(dx * dx + dy * dy)
                val mask = when {
                    distance <= 0.72f -> 1f
                    distance >= 1f -> 0f
                    else -> {
                        val t = ((1f - distance) / 0.28f).coerceIn(0f, 1f)
                        t * t * (3f - 2f * t)
                    }
                }
                val alpha = amount * mask * (Color.alpha(p) / 255f)
                if (alpha <= 0f) continue
                val original = originalRegion[i]
                fun mix(a: Int, b: Int) = (a + (b - a) * alpha).toInt().coerceIn(0, 255)
                originalRegion[i] = Color.rgb(
                    mix(Color.red(original), Color.red(p)),
                    mix(Color.green(original), Color.green(p)),
                    mix(Color.blue(original), Color.blue(p))
                )
            }
        }
        outputBitmap.setPixels(originalRegion, 0, width, left, top, width, height)
        return outputBitmap
    }
}
