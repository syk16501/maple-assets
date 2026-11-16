/* ===========================================================================
   ocr-worker.js —— EXP 数字识别 Worker（PaddleOCR PP-OCRv6_tiny_rec）
   ---------------------------------------------------------------------------
   本文件跑在 Web Worker 里，主线程只负责抓 ROI 的 ImageData 丢过来。
   为什么必须独立成 Worker：
     · 单次推理 9~13ms + 预处理 5~10ms，若在主线程跑，2 秒一次也会吃掉一帧
     · onnxruntime 的 wasm 初始化（编译 13.9MB wasm）会阻塞主线程数秒

   模型 rec.onnx 已把 6906 类分类头裁到 15 类（blank % . [ ] 0-9），
   体积 4.26MB → 2.13MB，且精度不降反升（详见 docs/OCR识别方案.md §3.5）。

   预处理必须与 Python 侧验证脚本保持一字不差，否则准确率会漂：
     找文字带（比背景亮 24 且饱和度 < 45）→ 裁带 → 双三次放大到字高 48
     → 灰度 → 对比度 ×1.10 → 3×3 十字 USM ×0.15 → BGR 归一化
   =========================================================================== */

/* ★ 必须用 ort.wasm.min.js（50KB），不能用 ort.all.min.js（819KB）：
   后者是「全 EP」构建，它加载的是 **jsep 版** wasm 胶水 + 二进制
   （ort-wasm-simd-threaded.jsep.wasm = 27.8MB，带 WebGPU/WebNN），
   我们只用 wasm EP，那 27.8MB 全是白付的带宽。
   实测只有这个纯 wasm 构建会去加载 non-jsep 的 13.9MB 版。 */
importScripts('ort.wasm.min.js');

/* 调试开关：想看内部状态（文字带/绿列/分段耗时）时改 true。
   正式版必须关掉 —— OCR 每 1.2 秒跑一次，控制台会被刷满。 */
var DBG = false;
function dbg(s) { if (DBG) console.log('[ocr] ' + s); }

/* CTC 输出类别（与模型裁剪时的 KEEP 顺序严格对应，改顺序必须重新裁模型） */
var LABELS = ['', '%', '.', '[', ']', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
var NCLS = LABELS.length;

var sess = null, inName = null, ready = false, basePath = '';

/* ---------------------------------------------------------------- 初始化 */
async function init(base) {
  /* ★ 必须用**绝对 URL**：onnxruntime 内部用动态 import() 加载 wasm 胶水文件，
     而 import() 的说明符必须以 './'、'/' 或协议开头 —— 传 'ocr/' 这种相对路径
     会抛 "Failed to resolve module specifier"，然后它 fallback 去找带 WebGPU 的
     jsep 版本（27.8MB），结果是又慢又 404。
     资源与 worker 同目录，所以直接按 worker 自身 URL 推目录。 */
  basePath = new URL('./', self.location.href).href;
  if (base && /^https?:/.test(base)) basePath = base;

  ort.env.wasm.wasmPaths = basePath;
  ort.env.wasm.numThreads = 1;      /* 不和游戏抢 CPU */
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;

  var t0 = performance.now();
  sess = await ort.InferenceSession.create(basePath + 'rec.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  });
  inName = sess.inputNames[0];
  /* 预热：首次 run 要建 wasm 内存池/算子缓存，实测第一次比后续慢 5~10 倍。
     不预热的话用户的第一拍读数会明显卡顿。 */
  try {
    await sess.run({ [inName]: new ort.Tensor('float32', new Float32Array(3 * 48 * 128), [1, 3, 48, 128]) });
  } catch (e) { /* 预热失败不影响正常使用 */ }
  ready = true;
  return Math.round(performance.now() - t0);
}

/* ------------------------------------------------------- 基础图像工具 */
function lumOf(data, i) { return data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114; }
function satOf(data, i) {
  var r = data[i], g = data[i + 1], b = data[i + 2];
  var mx = r > g ? (r > b ? r : b) : (g > b ? g : b);
  var mn = r < g ? (r < b ? r : b) : (g < b ? g : b);
  return mx - mn;
}

/* 文字带：比背景亮 24 以上、且饱和度 < 45（把绿色 [ ] % 和彩色 UI 排除掉）。
   背景不是取均值而是直方图中位数 —— 状态栏左半是深色背景，右半有进度条，
   均值会被进度条拉高，中位数才代表"底色"。 */
function findTextBand(buf, w, h) {
  var hist = new Uint32Array(256), i, x, y;
  for (i = 0, y = 0; y < h; y++) {
    for (x = 0; x < w; x++, i += 4) hist[(lumOf(buf, i) | 0)]++;
  }
  var total = w * h, half = total >> 1, acc = 0, bg = 128;
  for (i = 0; i < 256; i++) { acc += hist[i]; if (acc >= half) { bg = i; break; } }

  var thr = bg + 24;
  var rows = new Uint32Array(h);
  for (i = 0, y = 0; y < h; y++) {
    var n = 0;
    for (x = 0; x < w; x++, i += 4) {
      if (lumOf(buf, i) > thr && satOf(buf, i) < 45) n++;
    }
    rows[y] = n;
  }
  /* 文字行：有亮像素，但不能接近满行（满行是 UI 横线 / 进度条边框）。
     ★ 必须取「高度最大的连续段」，不能取「第一个遇到的段」。
       实测 2049×1152（150% 放大帧）在文字行上方还有 2 行模糊残影
       （亮像素 58 / 2），它会被当成一个 2 行高的文字带 → 直接被「高度 < 3」丢弃，
       然后报 no_band，整条 OCR 轨静默失效。
     ★ 搜索上限压到 ROI 的 75%：避开下方经验进度条。进度条每行有 100+ 亮像素，
       但它是**两条基准里都固定在下半部**的，压掉它就不用跟它比高度。
     ★ 段内允许 2 行空隙（数字上下不齐 / 轻微模糊）。 */
  var cap = w * 0.8, limit = Math.round(h * 0.75);
  var bestA = -1, bestB = -1, curA = -1, curB = -1, gap = 0, y, rowOk;
  for (y = 0; y < limit; y++) {
    rowOk = rows[y] > 0 && rows[y] <= cap;
    if (rowOk) {
      if (curA < 0) curA = y;
      curB = y; gap = 0;
    } else if (curA >= 0) {
      gap++;
      if (gap > 2) {
        if (bestA < 0 || (curB - curA) > (bestB - bestA)) { bestA = curA; bestB = curB; }
        curA = -1; curB = -1; gap = 0;
      }
    }
  }
  if (curA >= 0 && (bestA < 0 || (curB - curA) > (bestB - bestA))) { bestA = curA; bestB = curB; }
  if (bestA < 0 || bestB - bestA < 3) return null;
  return { y0: bestA, y1: bestB + 1, bg: bg };
}

/* 绿色分隔符：怀旧服的 [ ] % 是绿色，g 通道明显高于 r/b。
   按「列」统计而不是按块判断 —— 阈值抬高后 [ 整块会从 mask 里消失，
   只有逐列累计绿色像素才抓得住。 */
function greenStop(buf, w, h, y0, y1) {
  var need = Math.max(2, Math.round((y1 - y0) * 0.25));
  var x, y, i, n;
  for (x = 0; x < w; x++) {
    n = 0;
    for (y = y0, i = (y * w + x) * 4; y < y1; y++, i += w * 4) {
      var r = buf[i], g = buf[i + 1], b = buf[i + 2];
      if (g > r + 18 && g > b + 18 && g > 55) n++;
    }
    if (n >= need) return x;
  }
  return -1;
}

/* 双三次插值放大（Catmull-Rom 核），对应 Python 侧的 Image.BICUBIC。
   不用 canvas drawImage：Worker 里 OffscreenCanvas 的 smoothingQuality
   在各浏览器实现不一致，实测同一张图能差出 1px 笔画宽度。 */
function cubicKernel(t) {
  var a = -0.5, t2 = t * t, t3 = t2 * t;
  if (t <= 1) return (a + 2) * t3 - (a + 3) * t2 + 1;
  if (t < 2) return a * t3 - 5 * a * t2 + 8 * a * t - 4 * a;
  return 0;
}

function resizeCubic(src, sw, sh, dw, dh) {
  var out = new Float32Array(dw * dh);
  var xr = sw / dw, yr = sh / dh;
  var x, y, m, n, fx, fy, acc, wsum, wx, wy, sx, sy;
  for (y = 0; y < dh; y++) {
    sy = (y + 0.5) * yr - 0.5;
    var iy = Math.floor(sy);
    for (x = 0; x < dw; x++) {
      sx = (x + 0.5) * xr - 0.5;
      var ix = Math.floor(sx);
      acc = 0; wsum = 0;
      for (m = -1; m <= 2; m++) {
        var py = iy + m; if (py < 0) py = 0; if (py >= sh) py = sh - 1;
        wy = cubicKernel(sy - (iy + m));
        if (wy === 0) continue;
        for (n = -1; n <= 2; n++) {
          var px = ix + n; if (px < 0) px = 0; if (px >= sw) px = sw - 1;
          wx = cubicKernel(sx - (ix + n));
          if (wx === 0) continue;
          wsum += wx * wy;
          acc += wx * wy * src[py * sw + px];
        }
      }
      out[y * dw + x] = wsum !== 0 ? acc / wsum : 0;
    }
  }
  return out;
}

/* ------------------------------------------------------------ 单段预处理 */
/* 返回 Float32Array（已按 [1,3,48,W] 的 BGR 归一化展开）+ 宽度 */
function prepSegment(buf, w, h, x0, x1, band, targetH, contrast, sharpen) {
  var bw = x1 - x0;
  if (bw < 3) return null;
  var bh = band.y1 - band.y0;
  if (bh < 3) return null;

  /* 1. 取文字带的灰度 */
  var src = new Float32Array(bw * bh);
  var x, y, i, o = 0;
  for (y = band.y0; y < band.y1; y++) {
    i = (y * w + x0) * 4;
    for (x = 0; x < bw; x++, i += 4) src[o++] = lumOf(buf, i);
  }

  /* 2. 放大到目标字高（上限 12 倍，和枫记一致） */
  var sc = targetH / bh;
  if (sc < 1) sc = 1; if (sc > 12) sc = 12;
  var dw = Math.max(1, Math.round(bw * sc)), dh = Math.max(1, Math.round(bh * sc));
  if (dw > 900) { dw = 900; }          /* 超长会拖慢推理且没必要 */
  var big = resizeCubic(src, bw, bh, dw, dh);

  /* 3. 对比度 */
  for (i = 0; i < big.length; i++) big[i] = 128 + (big[i] - 128) * contrast;
  if (sharpen > 0) {
    var tmp = new Float32Array(big.length);
    var cx, cy;
    for (cy = 0; cy < dh; cy++) {
      for (cx = 0; cx < dw; cx++) {
        var up = (cy > 0 ? cy - 1 : cy) * dw + cx;
        var dn = (cy < dh - 1 ? cy + 1 : cy) * dw + cx;
        var lf = cy * dw + (cx > 0 ? cx - 1 : cx);
        var rt = cy * dw + (cx < dw - 1 ? cx + 1 : cx);
        var idx = cy * dw + cx;
        var mean4 = (big[up] + big[dn] + big[lf] + big[rt]) / 4;
        tmp[idx] = big[idx] + (big[idx] - mean4) * sharpen;
      }
    }
    big = tmp;
  }

  /* 4. ★ 模型输入高度**固定 48**。第二档「放大到 64」再缩回 48 是故意的二次采样
     （Python 侧 to_input() 正是这么做），实测能救回少数「8 被读成 6」的帧。
     少这一步会直接抛 "Got invalid dimensions for input: x ... Got: 64 Expected: 48"。 */
  if (dh !== 48) {
    var dw2 = Math.max(1, Math.round(dw * 48 / dh));
    if (dw2 > 900) dw2 = 900;
    big = resizeCubic(big, dw, dh, dw2, 48);
    dw = dw2; dh = 48;
  }

  /* 5. 归一化：BGR、(v/255-0.5)/0.5，展平成 CHW */
  var n = dw * dh;
  var out = new Float32Array(3 * n);
  for (var k = 0; k < n; k++) {
    var v = big[k];
    if (v < 0) v = 0; if (v > 255) v = 255;
    var t = (v / 255 - 0.5) / 0.5;
    out[k] = t;            /* R（灰度三通道相同，BGR 顺序只是形式） */
    out[n + k] = t;
    out[2 * n + k] = t;
  }
  return { data: out, w: dw, h: dh };
}

/* -------------------------------------------------------------- CTC 解码 */
function ctcDecode(logits, T, C) {
  var out = '', prev = -1, sum = 0, cnt = 0;
  for (var t = 0; t < T; t++) {
    var base = t * C, best = 0, bv = logits[base];
    for (var c = 1; c < C; c++) {
      var v = logits[base + c];
      if (v > bv) { bv = v; best = c; }
    }
    if (best !== 0 && best !== prev) {
      out += LABELS[best] || '';
      sum += bv; cnt++;
    }
    prev = best;
  }
  return { text: out, conf: cnt ? sum / cnt : 0 };
}

function digitsOf(s) {
  var t = s.replace(/[Oo]/g, '0').replace(/[Il|]/g, '1');
  var r = '';
  for (var i = 0; i < t.length; i++) {
    var c = t.charAt(i);
    if ((c >= '0' && c <= '9') || c === '.') r += c;
  }
  return r;
}

/* ------------------------------------------------------------ 一次完整识别 */
var ATTEMPTS = [
  { h: 48, ct: 1.10, sh: 0.00 },
  { h: 64, ct: 1.16, sh: 0.15 }
];

async function readOnce(buf, w, h) {
  var band = findTextBand(buf, w, h);
  if (!band) return { err: 'no_band' };

  var gx = greenStop(buf, w, h, band.y0, band.y1);
  dbg('roi=' + w + 'x' + h + ' band=' + band.y0 + '..' + band.y1 + ' bg=' + band.bg + ' gx=' + gx);

  /* ★ num 段左界：必须跳过左侧的 EXP 标签区，不能从 0 起切。
     ROI 是按帧宽定比裁的，标签布局在 ROI 内是恒定的（实测三张基准图）：
       EXP 标签右缘 ≈ 0.404w   （1366→72/178、2049→108/267，完全一致）
       数字左缘     ≈ 0.472w   （1366→84/178、2049→120/267，只差 1~2px 模糊外溢）
     取 0.46w 落在这条空档里，两边各留一点余量。
     ★ 从 0 起切的后果（2026-10-18 实测）：糊帧上 EXP 的残留笔画会被读成数字，
       `1422051` → `81422051` / `91422051`。**多一位比少一位更危险** ——
       81,422,051 是一个"合法"的经验值，主线程的单调性校验会直接采信，
       而少一位只会被 conf 或单调性挡掉。
     ★ 也不能切到 0.48w（实测）：那会啃掉首位数字，1422051 → 422051。 */
  var numX0 = Math.round(w * 0.46);
  /* 右界：绿色 [ 位置最准；找不到绿色（糊帧上 [ 整体褪色）时退 0.695w ——
     这是三张基准图实测 [ 位置的一致比例（0.691 / 0.699 / 0.697）。
     历史 bug：这里原来退 0.62w，直接把末位数字切掉（1422051 → 14220）。 */
  var numX1 = (gx > 4) ? Math.min(w, gx + 3) : Math.round(w * 0.695);
  if (numX1 <= numX0 + 4) numX1 = w;      /* 兜底：切出来太窄说明定位失败，整段喂 */

  /* 百分比段左边界要往左退 3px：视频编码（captureStream / 屏幕捕获 →
     编码 → 解码）会让绿色 [ 的位置比静态图偏 1~2px，按 gx+1 起切实测会把
     首位数字切掉（76.99 → 6.99）。多带进来的 [ 无所谓 —— digitsOf 只留数字和点。
     找不到绿色时退 0.70w（实测比 0.62w 好：0.62w 会把数字尾和 [ 一起卷进来，
     读成 `5158.74`）。 */
  var pctX0 = (gx > 4) ? Math.max(0, gx - 3) : Math.round(w * 0.70);
  var segs = [
    { key: 'num', x0: numX0, x1: numX1 },
    { key: 'pct', x0: pctX0, x1: w }
  ];

  var res = {};
  for (var si = 0; si < segs.length; si++) {
    var sg = segs[si];
    var best = null;
    for (var ai = 0; ai < ATTEMPTS.length; ai++) {
      var A = ATTEMPTS[ai];
      var _t0 = performance.now();
      var p = prepSegment(buf, w, h, sg.x0, sg.x1, band, A.h, A.ct, A.sh);
      if (!p) continue;
      var _t1 = performance.now();
      var tensor = new ort.Tensor('float32', p.data, [1, 3, p.h, p.w]);
      var out;
      try { out = await sess.run({ [inName]: tensor }); }
      catch (e) { return { err: 'run:' + e.message }; }
      var _t2 = performance.now();
      if (A === ATTEMPTS[0]) dbg(sg.key + ' prep=' + (_t1 - _t0).toFixed(1)
        + ' run=' + (_t2 - _t1).toFixed(1) + ' w=' + p.w + 'h=' + p.h);
      var first = out[sess.outputNames[0]];
      var dims = first.dims;                    /* [1, T, C] */
      var T = dims[1], C = dims[2];
      var dec = ctcDecode(first.data, T, C);
      dec.digits = digitsOf(dec.text);
      if (!best || dec.conf > best.conf) best = dec;
      /* 首档就够自信就不再跑第二档 —— 第二档要额外做一次 64→48 的二次重采样
         + 一次推理，实测占掉整轮一半时间。清晰帧 conf 稳定在 0.95 以上，
         只有糊帧才会掉下来需要第二档去救。 */
      if (best.conf >= 0.93) break;
    }
    res[sg.key] = best;
  }

  /* ★ 物理上限闸门：糊帧上模型仍可能吐出一个「格式合法但量级离谱」的结果
     （EXP 十几位、百分比 > 100）。它比「读不出」更危险 —— 主线程的单调性
     校验只挡回落，不挡暴涨，一旦采信就会把经验时速算到天上。
     经验值上限 8 位（99,999,999），百分比 0~100 且最多 3 位整数 + 2 位小数。 */
  var num = res.num, pct = res.pct;
  if (num && num.digits.length > 8) { num.bad = num.digits; num.digits = ''; num.conf = 0; }
  if (pct && pct.digits) {
    var pv = parseFloat(pct.digits);
    if (!isFinite(pv) || pv > 100 || pct.digits.length > 6) { pct.digits = ''; pct.conf = 0; }
  }
  return res;
}

/* ---------------------------------------------------------------- 消息循环 */
self.onmessage = async function (e) {
  var m = e.data;

  if (m.cmd === 'init') {
    try {
      var ms = await init(m.base);
      self.postMessage({ type: 'ready', ms: ms });
    } catch (err) {
      self.postMessage({ type: 'fail', stage: 'init', msg: String(err && err.message || err) });
    }
    return;
  }

  if (m.cmd === 'read') {
    if (!ready) { self.postMessage({ type: 'result', id: m.id, err: 'not_ready' }); return; }
    var t0 = performance.now();
    try {
      var r = await readOnce(new Uint8ClampedArray(m.data), m.w, m.h);
      if (r.err) {
        self.postMessage({ type: 'result', id: m.id, err: r.err, ms: performance.now() - t0 });
        return;
      }
      var num = r.num, pct = r.pct;
      self.postMessage({
        type: 'result', id: m.id, ms: performance.now() - t0,
        cur: num ? num.digits : '',
        curConf: num ? num.conf : 0,
        curRaw: num ? num.text : '',
        pct: pct ? pct.digits : '',
        pctConf: pct ? pct.conf : 0,
        pctRaw: pct ? pct.text : ''
      });
    } catch (err) {
      self.postMessage({ type: 'result', id: m.id, err: 'read:' + String(err && err.message || err) });
    }
    return;
  }
};
