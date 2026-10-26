package com.naqaa.ai

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.net.Uri
import android.content.ContentValues
import android.os.Build
import android.provider.MediaStore
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
    var status by remember { mutableStateOf("اختر صورة للبدء") }
    var strength by remember { mutableFloatStateOf(0.8f) }

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
                        Text("تحسين الصور واستعادة الوضوح", style = MaterialTheme.typography.bodyMedium,
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
                        Text("النتيجة حدة محلية تجريبية وليست استعادة بالذكاء الاصطناعي.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.secondary,
                            modifier = Modifier.padding(top = 8.dp))
                    }
                    Spacer(Modifier.height(18.dp))
                    Text("قوة التحسين: ${(strength * 100).toInt()}%",
                        modifier = Modifier.align(Alignment.Start))
                    Slider(value = strength, onValueChange = { strength = it })
                    Button(
                        modifier = Modifier.fillMaxWidth().height(52.dp),
                        enabled = !busy,
                        onClick = {
                            val source = original ?: return@Button
                            scope.launch {
                                busy = true
                                status = "جارٍ تحسين الصورة محليًا..."
                                enhanced = withContext(Dispatchers.Default) {
                                    LocalEnhancer.enhance(source, strength)
                                }
                                status = "اكتمل التحسين المحلي التجريبي"
                                busy = false
                            }
                        }
                    ) {
                        Icon(Icons.Default.AutoFixHigh, contentDescription = null)
                        Spacer(Modifier.width(8.dp))
                        Text("تحسين محلي تجريبي")
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
                Text("نسخة أولية • 0.1.0", style = MaterialTheme.typography.labelSmall,
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

/**
 * Simple on-device unsharp-mask prototype.
 * This is not a trained face-restoration model and cannot recover lost detail.
 */
private object LocalEnhancer {
    fun enhance(src: Bitmap, amount: Float): Bitmap {
        val maxSide = 1600
        val scale = minOf(1f, maxSide.toFloat() / maxOf(src.width, src.height))
        val input = if (scale < 1f) Bitmap.createScaledBitmap(
            src, (src.width * scale).toInt().coerceAtLeast(1),
            (src.height * scale).toInt().coerceAtLeast(1), true
        ) else src

        val w = input.width
        val h = input.height
        val pixels = IntArray(w * h)
        input.getPixels(pixels, 0, w, 0, 0, w, h)
        val out = pixels.copyOf()
        // A stronger 3x3 Gaussian unsharp mask. Higher strength makes edge/detail
        // differences more visible while retaining the original image dimensions.
        val a = amount.coerceIn(0f, 1f) * 3.0f

        for (y in 1 until h - 1) {
            for (x in 1 until w - 1) {
                val i = y * w + x
                val center = pixels[i]
                val topLeft = pixels[i - w - 1]
                val top = pixels[i - w]
                val topRight = pixels[i - w + 1]
                val left = pixels[i - 1]
                val right = pixels[i + 1]
                val bottomLeft = pixels[i + w - 1]
                val bottom = pixels[i + w]
                val bottomRight = pixels[i + w + 1]

                fun channel(c: (Int) -> Int): Int {
                    val base = c(center)
                    val weightedBlur = (
                        c(topLeft) + 2 * c(top) + c(topRight) +
                        2 * c(left) + 4 * base + 2 * c(right) +
                        c(bottomLeft) + 2 * c(bottom) + c(bottomRight)
                    ) / 16f
                    return (base + a * (base - weightedBlur)).toInt().coerceIn(0, 255)
                }
                out[i] = Color.argb(
                    Color.alpha(center),
                    channel { Color.red(it) },
                    channel { Color.green(it) },
                    channel { Color.blue(it) }
                )
            }
        }
        return Bitmap.createBitmap(out, w, h, Bitmap.Config.ARGB_8888)
    }
}
