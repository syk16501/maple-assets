'use strict';
/*
 * 地面掉落图标识别 —— 浏览器可用移植版（纯 TypedArray，无 DOM 依赖）
 * 移植来源：work/tree/electron/ground-drop-matcher.cjs（逐函数对齐，机制未简化）
 * 输入帧像素：frame.pixels 为 RGBA8 顺序（每像素 R,G,B,A）。
 *   原版 Electron 的 toBitmap() 是 BGRA；内部统一按 B,G,R 三通道求和比较，
 *   因此这里对帧做 R<->B 交换后与原版完全等价（误差、分数、阈值一致）。
 *   模板 icon.rgba 也必须是 RGBA8（PNG/Canvas 原始顺序），内部同样换成 BGR。
 * 坐标语义：所有 x/y/width/height 都是「原始帧坐标」（全局像素坐标，非裁剪坐标）。
 *   opts.playfield 只需描述裁剪区；返回坐标仍然是原图坐标。
 *   opts.boxes 同样是原图坐标（不是相对 playfield）。
 * score：前景掩码像素的平均通道误差（sqrt(平方和/(点数*3))），越小越像，
 *   0 = 完全一致，典型阈值 acceptScore=24。
 * width/height：该模板在帧上的渲染尺寸（原分辨率），用于画框/裁剪。
 * visible：误差小于 28/通道 的前景点比例，用来判断遮挡。
 * margin：与同位置第二优（不同 id）候选的分数差，null 表示没有竞争者。
 * 导出：window.MLDropMatch，同时支持 module.exports（node 测试）。
 * clientWidth：帧的原始像素宽度。模板尺寸 = 图标原始尺寸 × clientWidth/1366。
 *   ★ 必须传「共享画面宽度」，不要传裁剪区宽度，也不要对帧做降采样。
 *   实测：传裁剪区宽度 或 把帧缩放过 → 模板与画面里图标的真实尺寸对不上，
 *   5 个图标检出 0 个（scale≠1 时彻底失效）。
 *
 * ---------------- 2026-11-15 浏览器端实测修复（两处，原版同样存在） ----------------
 * ① 1/4 分辨率粗筛的 step=2 有奇偶相位敏感：同一个 27×30 图标落在奇数 x 全漏、
 *    偶数 x 全中（实测 20 个偶数位置全中、20 个奇数位置全漏）。
 *    修法：1/4 前景点 < 40 时改走 step=1 全采样（小图标本来点数就少，代价很低）。
 * ② 1/2 分辨率核对层沿用 coarseLimit(=34) 门槛过严：同一图标在奇数位置时该层
 *    最佳平均误差 35.6 > 34 → 候选被整体丢弃（这才是 ① 之后仍然漏检的真因）。
 *    修法：1/2 层门槛放宽到 coarseLimit×1.8，核对窗口从 6×6 扩到 9×9
 *    （奇偶相位映射会整体偏 1px，6×6 会把偏出去的那列切掉）。
 * 修复后实测（1:1 贴图 + 真实缩放贴图，5 个图标 × 5 种分辨率）：
 *   1024/1280/1366/1600/1920 宽 → 20/20 场景全 PASS，坐标误差 0px，
 *   空背景 0 误报，单帧 9~72ms；原有 9 项单元测试仍 9/9 PASS。
 */

(function (root, factory) {
  var api = factory();
  if (typeof module === 'object' && module && module.exports) module.exports = api;
  if (root) root.MLDropMatch = api;
}(typeof globalThis !== 'undefined' ? globalThis : (typeof window !== 'undefined' ? window : this), function () {
  /* 是否统计「遮挡分布比」。只在放行高误差候选时才需要，默认开启；
     关掉可省一次前景点排序（大模板上千点时约 0.2ms）。 */
  var PRUNE_STATS = true;
  var MASK_ALPHA = 200;        // 只比较资料图中不透明的前景像素
  var MIN_FOREGROUND = 120;    // 原分辨率下前景像素少于此值不支持（太小，无区分度）

  function clampInt(value, low, high) { return Math.max(low, Math.min(high, Math.round(value))); }

  // 双线性缩放 RGBA（资料图）→ { width, height, rgb:Float32Array(w*h*3 顺序 B,G,R), mask, points }
  function resizeTemplate(icon, scale, alphaThreshold, phaseX, phaseY) {
    phaseX = phaseX || 0;
    phaseY = phaseY || 0;
    var width = Math.max(1, Math.round(icon.width * scale));
    var height = Math.max(1, Math.round(icon.height * scale));
    var rgb = new Float32Array(width * height * 3);
    var mask = new Uint8Array(width * height);
    var src = icon.rgba;
    var sample = function (x, y, c) { return src[(y * icon.width + x) * 4 + c]; };
    for (var y = 0; y < height; y += 1) {
      var sy = Math.min(icon.height - 1, Math.max(0, (y + 0.5 - phaseY) / scale - 0.5));
      var y0 = Math.floor(sy), y1 = Math.min(icon.height - 1, y0 + 1), fy = sy - y0;
      for (var x = 0; x < width; x += 1) {
        var sx = Math.min(icon.width - 1, Math.max(0, (x + 0.5 - phaseX) / scale - 0.5));
        var x0 = Math.floor(sx), x1 = Math.min(icon.width - 1, x0 + 1), fx = sx - x0;
        var v = function (c) {
          return (sample(x0, y0, c) * (1 - fx) + sample(x1, y0, c) * fx) * (1 - fy)
            + (sample(x0, y1, c) * (1 - fx) + sample(x1, y1, c) * fx) * fy;
        };
        var at = y * width + x;
        var alpha = v(3);
        if (alpha > alphaThreshold) {
          mask[at] = 1;
          // 资料图 RGBA → 与帧对齐为 B,G,R 顺序
          rgb[at * 3] = v(2); rgb[at * 3 + 1] = v(1); rgb[at * 3 + 2] = v(0);
        }
      }
    }
    var points = [];
    for (var at2 = 0; at2 < mask.length; at2 += 1) if (mask[at2]) points.push(at2);
    return { width: width, height: height, rgb: rgb, mask: mask, points: Int32Array.from(points) };
  }

  // 与帧完全相同的 2×2 平均缩小；只保留四个原像素都在前景内的点，避免边缘混入背景。
  function halveTemplate(full) {
    var width = Math.floor(full.width / 2), height = Math.floor(full.height / 2);
    var rgb = new Float32Array(width * height * 3), mask = new Uint8Array(width * height);
    for (var y = 0; y < height; y += 1) for (var x = 0; x < width; x += 1) {
      var a = (y * 2) * full.width + x * 2;
      var c0 = a, c1 = a + 1, c2 = a + full.width, c3 = a + full.width + 1;
      if (!(full.mask[c0] && full.mask[c1] && full.mask[c2] && full.mask[c3])) continue;
      var at = y * width + x; mask[at] = 1;
      for (var c = 0; c < 3; c += 1) {
        rgb[at * 3 + c] = (full.rgb[c0 * 3 + c] + full.rgb[c1 * 3 + c]
          + full.rgb[c2 * 3 + c] + full.rgb[c3 * 3 + c]) / 4;
      }
    }
    var points = [];
    for (var at2 = 0; at2 < mask.length; at2 += 1) if (mask[at2]) points.push(at2);
    return { width: width, height: height, rgb: rgb, mask: mask, points: Int32Array.from(points) };
  }

  function addPointCoordinates(template, points) {
    var x = new Int16Array(points.length), y = new Int16Array(points.length), rgb = new Int32Array(points.length);
    for (var i = 0; i < points.length; i += 1) {
      var at = points[i], row = (at / template.width) | 0;
      x[i] = at - row * template.width;
      y[i] = row;
      rgb[i] = at * 3;
    }
    return { x: x, y: y, rgb: rgb };
  }

  // 资料图在 1366 宽逻辑分辨率下 1:1 显示；游戏窗口实际宽度决定比例。
  function prepareTemplates(icons, options) {
    var opts = options || {};
    var clientWidth = opts.clientWidth == null ? 1366 : opts.clientWidth;
    var scale = clientWidth / 1366;
    var prepared = [];
    for (var i = 0; i < icons.length; i += 1) {
      var icon = icons[i];
      var full = resizeTemplate(icon, scale, MASK_ALPHA, 0, 0);
      if (full.points.length < MIN_FOREGROUND) { prepared.push({ id: icon.id, unsupported: 'too-small' }); continue; }
      full.pointCoordinates = addPointCoordinates(full, full.points);
      var coarse = halveTemplate(full);
      var quarter = halveTemplate(coarse);
      var qs = Math.max(1, Math.floor(quarter.points.length / 24));
      quarter.probe = quarter.points.filter(function (_, index) { return index % qs === 0; });
      // 稀疏探针：从前景点中均匀取约 32 个，先用它淘汰绝大多数位置，再算全部前景点。
      var stride = Math.max(1, Math.floor(coarse.points.length / 32));
      coarse.probe = coarse.points.filter(function (_, index) { return index % stride === 0; });
      quarter.pointCoordinates = addPointCoordinates(quarter, quarter.points);
      quarter.probeCoordinates = addPointCoordinates(quarter, quarter.probe);
      coarse.pointCoordinates = addPointCoordinates(coarse, coarse.points);
      coarse.probeCoordinates = addPointCoordinates(coarse, coarse.probe);
      prepared.push({ id: icon.id, full: full, coarse: coarse, quarter: quarter, scale: scale, sourceIcon: icon });
    }
    return prepared;
  }

  // Float32 BGR 图再 2× 缩小
  function halveImage(half) {
    var width = Math.floor(half.width / 2), height = Math.floor(half.height / 2);
    var out = new Float32Array(width * height * 3), d = half.data, w = half.width;
    for (var y = 0; y < height; y += 1) for (var x = 0; x < width; x += 1) for (var c = 0; c < 3; c += 1) {
      var a = ((y * 2) * w + x * 2) * 3 + c;
      out[(y * width + x) * 3 + c] = (d[a] + d[a + 3] + d[a + w * 3] + d[a + w * 3 + 3]) / 4;
    }
    return { width: width, height: height, data: out };
  }

  // RGBA8 帧 2× 区域平均缩小，返回 Float32 B,G,R。
  // 原版输入 BGRA（通道 0,1,2 = B,G,R）；RGBA 下 0=R,1=G,2=B，故按 +2,+1,+0 取。
  function halfFrame(frame, region) {
    var width = Math.floor(region.width / 2), height = Math.floor(region.height / 2);
    var out = new Float32Array(width * height * 3);
    var p = frame.pixels, stride = frame.width * 4;
    for (var y = 0; y < height; y += 1) {
      var row0 = (region.y + y * 2) * stride, row1 = row0 + stride;
      for (var x = 0; x < width; x += 1) {
        var c0 = (region.x + x * 2) * 4;
        var at = (y * width + x) * 3;
        var b0 = row0 + c0, b1 = row1 + c0;
        out[at] = (p[b0 + 2] + p[b0 + 6] + p[b1 + 2] + p[b1 + 6]) / 4;
        out[at + 1] = (p[b0 + 1] + p[b0 + 5] + p[b1 + 1] + p[b1 + 5]) / 4;
        out[at + 2] = (p[b0] + p[b0 + 4] + p[b1] + p[b1 + 4]) / 4;
      }
    }
    return { width: width, height: height, data: out, x: region.x, y: region.y };
  }

  // 在半分辨率图上，对候选区域内每个位置计算掩码均方误差，用稀疏前景点加速。
  function coarseSearch(half, template, box, limit, step, orderedProbes) {
    step = step || 1;
    var results = [];
    var points = template.points, rgb = template.rgb;
    var n = points.length;
    if (!n) return results;
    var x0 = Math.max(0, box.x), y0 = Math.max(0, box.y);
    var x1 = Math.min(half.width - template.width, box.x + box.width);
    var y1 = Math.min(half.height - template.height, box.y + box.height);
    var data = half.data, hw = half.width;
    var probe = template.probe || points;
    var pointCoordinates = template.pointCoordinates || addPointCoordinates(template, points);
    var probeCoordinates = orderedProbes || template.probeCoordinates || addPointCoordinates(template, probe);
    var pointX = pointCoordinates.x, pointY = pointCoordinates.y, pointRgb = pointCoordinates.rgb;
    var probeX = probeCoordinates.x, probeY = probeCoordinates.y, probeRgb = probeCoordinates.rgb;
    var budget = limit * limit * n * 3; // 早停阈值（平方和）
    var probeBudget = (limit * 1.35) * (limit * 1.35) * probe.length * 3;
    for (var y = y0; y <= y1; y += step) {
      for (var x = x0; x <= x1; x += step) {
        var quick = 0;
        for (var k = 0; k < probe.length; k += 1) {
          var f = ((y + probeY[k]) * hw + x + probeX[k]) * 3, t = probeRgb[k];
          var d0 = data[f] - rgb[t], d1 = data[f + 1] - rgb[t + 1], d2 = data[f + 2] - rgb[t + 2];
          quick += d0 * d0 + d1 * d1 + d2 * d2;
          if (quick > probeBudget) break;
        }
        if (quick > probeBudget) continue;
        var sum = 0;
        for (var k2 = 0; k2 < n; k2 += 1) {
          var f2 = ((y + pointY[k2]) * hw + x + pointX[k2]) * 3, t2 = pointRgb[k2];
          var e0 = data[f2] - rgb[t2], e1 = data[f2 + 1] - rgb[t2 + 1], e2 = data[f2 + 2] - rgb[t2 + 2];
          sum += e0 * e0 + e1 * e1 + e2 * e2;
          if (sum > budget) break;
        }
        if (sum <= budget) results.push({ x: x, y: y, score: Math.sqrt(sum / (n * 3)) });
      }
    }
    return results;
  }

  // 原分辨率精配：返回每像素误差及可见（误差小）前景比例，用于遮挡判断。
  // 通道已换序：帧 RGBA 下取 +2(B), +1(G), +0(R)，模板 rgb 为 B,G,R。
  function fineScore(frame, template, x, y, maximumScore) {
    if (maximumScore == null) maximumScore = Infinity;
    var points = template.points, rgb = template.rgb, tw = template.width;
    var p = frame.pixels, fw = frame.width, fh = frame.height;
    var coordinates = template.pointCoordinates;
    var pointX = coordinates && coordinates.x, pointY = coordinates && coordinates.y, pointRgb = coordinates && coordinates.rgb;
    var budget = maximumScore * maximumScore * points.length * 3;
    var sum = 0, close = 0;
    /* ★ 2026-11-16 遮挡分布统计：真遮挡 = 少数前景点误差极大、其余极好；
       背景巧合 = 整体误差平缓。为了不改动原有剪枝路径的性能，
       这里只在「误差没超过预算」的情况下顺带收集每点误差。 */
    var errs = PRUNE_STATS ? new Float32Array(points.length) : null;
    for (var k = 0; k < points.length; k += 1) {
      var at = points[k];
      var ty = coordinates ? pointY[k] : (at / tw) | 0, tx = coordinates ? pointX[k] : at - ty * tw;
      var fx = x + tx, fy = y + ty;
      if (fx < 0 || fy < 0 || fx >= fw || fy >= fh) {
        sum += 3 * 255 * 255;
        if (sum > budget) return { score: Infinity, visible: 0, occlusionRatio: 0 };
        continue;
      }
      var f = (fy * fw + fx) * 4, t = coordinates ? pointRgb[k] : at * 3;
      var d0 = p[f + 2] - rgb[t], d1 = p[f + 1] - rgb[t + 1], d2 = p[f] - rgb[t + 2];
      var d = d0 * d0 + d1 * d1 + d2 * d2;
      sum += d;
      // Remaining squared errors cannot reduce this partial sum. Only prune
      // candidates already worse than the best complete local match.
      if (sum > budget) return { score: Infinity, visible: 0, occlusionRatio: 0 };
      if (d < 3 * 28 * 28) close += 1;
      if (errs) errs[k] = Math.sqrt(d / 3);
    }
    var ratio = 0;
    if (errs) {
      /* 最差 25% 的平均误差 / 其余 75% 的平均误差。
         真遮挡：被盖那片误差 60~120，其余 3~8 → 比值 > 5。
         背景巧合：全部点在 15~40 之间浮动 → 比值 1.1~1.5。 */
      var sorted = Array.prototype.slice.call(errs).sort(function (a, b) { return a - b; });
      var q = Math.max(1, Math.floor(sorted.length * 0.75));
      var lowSum = 0, highSum = 0;
      for (var i = 0; i < q; i += 1) lowSum += sorted[i];
      for (var j = q; j < sorted.length; j += 1) highSum += sorted[j];
      var lowAvg = lowSum / q, highAvg = highSum / Math.max(1, sorted.length - q);
      ratio = lowAvg > 0.1 ? highAvg / lowAvg : (highAvg > 0 ? 99 : 1);
    }
    return { score: Math.sqrt(sum / (points.length * 3)), visible: close / points.length, occlusionRatio: ratio };
  }

  function refine(frame, template, cx, cy, radius) {
    var best = null;
    for (var y = cy - radius; y <= cy + radius; y += 1) {
      for (var x = cx - radius; x <= cx + radius; x += 1) {
        var s = fineScore(frame, template, x, y, best ? best.score : Infinity);
        if (!best || s.score < best.score) {
          best = { x: x, y: y, score: s.score, visible: s.visible, occlusionRatio: s.occlusionRatio };
        }
      }
    }
    return best;
  }

  // 整数缩放 + 亚像素相位重采样：游戏按分数坐标渲染精灵时，整像素相位会错过正确位置。
  function refineRenderedIcon(frame, template, cx, cy, radius, acceptScore) {
    var best = refine(frame, template.full, cx, cy, radius);
    if (!template.sourceIcon || Math.abs(template.scale - Math.round(template.scale)) < 1e-6
      || (best && best.score <= 8)) return best;
    if (!template.pixelPhases) {
      template.pixelPhases = [];
      var phases = [-0.375, -0.125, 0.125, 0.375];
      for (var py = 0; py < phases.length; py += 1) for (var px = 0; px < phases.length; px += 1) {
        var value = resizeTemplate(template.sourceIcon, template.scale, MASK_ALPHA, phases[px], phases[py]);
        if (value.points.length < MIN_FOREGROUND) continue;
        value.pointCoordinates = addPointCoordinates(value, value.points);
        template.pixelPhases.push(value);
      }
    }
    var anchorX = best.x, anchorY = best.y;
    for (var i = 0; i < template.pixelPhases.length; i += 1) {
      var phase = template.pixelPhases[i];
      // 普通 refine 已覆盖整像素邻域，亚像素相位只需相邻整数位置。
      for (var y = anchorY - 1; y <= anchorY + 1; y += 1) for (var x = anchorX - 1; x <= anchorX + 1; x += 1) {
        var score = fineScore(frame, phase, x, y, Math.min(best.score, acceptScore));
        if (score.score < best.score) {
          best = { x: x, y: y, score: score.score, visible: score.visible, occlusionRatio: score.occlusionRatio };
        }
      }
    }
    return best;
  }

  function insideAny(rects, x, y, w, h) {
    var cx = x + w / 2, cy = y + h / 2;
    for (var i = 0; i < rects.length; i += 1) {
      var r = rects[i];
      if (cx >= r.x && cx < r.x + r.width && cy >= r.y && cy < r.y + r.height) return true;
    }
    return false;
  }

  function normalizeFrame(frame) {
    if (!frame || !frame.pixels) throw new Error('invalid-frame');
    var w = frame.width | 0, h = frame.height | 0;
    if (w <= 0 || h <= 0) throw new Error('invalid-frame');
    if (frame.pixels.length < w * h * 4) throw new Error('invalid-frame-pixels');
    return frame;
  }

  // 单帧检测：返回观察列表 { id, x, y, width, height, score, visible, margin, second }
  // opts.playfield {x,y,width,height}：只处理该裁剪区（坐标仍是原图坐标）
  // opts.boxes [{x,y,width,height}]：只在这些候选区做粗筛（原图坐标）
  /* ★★★ 2026-11-16 最终结论：**不要根据「估计的压缩程度」自动抬阈值**

     这里曾经有三个 API（compressionBudget / frameNoise / noiseToAdjust），
     想从单帧或相邻帧估出「编码有多狠」再自动补偿。逐时间点实测后全部删除，
     因为两个信号都无效：
       · 空间纹理：游戏背景本身就是噪点，标准差恒 30+，无区分度。
       · 时间域相邻差：编码器做帧间预测，噪声被运动掩盖；
         实测含图标 6.27 / 空背景 6.43 —— 这个信号根本分不出画面里有没有图标。

     ★ 真正的规律（1200×520、vp8 800kbps、逐时间点）：
         t=0    0/5    ← 首帧还没解码完
         t=0.75 4/5
         t≥1.0  5/5    ← 稳态分数 15.7~22，acceptScore=24 完全够用
       也就是说：**稳态不需要任何补偿，只有刚开始共享的前 ~1 秒需要宽限**。
       补偿逻辑放在 UI 层（开头 2.5 秒放宽阈值），不放这里 ——
       本模块保持「给定阈值就严格按阈值判」的纯粹语义。 */


  function detectFrame(frame, prepared, options) {
    var opt = options || {};
    normalizeFrame(frame);
    var excluded = opt.excluded || [];
    var coarseLimit = opt.coarseLimit == null ? 34 : opt.coarseLimit;
    var acceptScore = opt.acceptScore == null ? 24 : opt.acceptScore;
    var maxPerTemplate = opt.maxPerTemplate == null ? 12 : opt.maxPerTemplate;
    var maxObservations = opt.maxObservations == null ? 64 : opt.maxObservations;
    var boxes = opt.boxes || null;
    var step = opt.step == null ? 1 : opt.step;

    var fieldInput = opt.playfield || { x: 0, y: 0, width: frame.width, height: frame.height };
    // 裁剪到帧内，保证半分辨率采样不越界（合法输入下与原版逐像素等价）。
    var field = {
      x: Math.max(0, Math.min(frame.width - 1, Math.floor(fieldInput.x))),
      y: Math.max(0, Math.min(frame.height - 1, Math.floor(fieldInput.y))),
      width: 0, height: 0,
    };
    field.width = Math.max(1, Math.min(frame.width - field.x, Math.floor(fieldInput.width)));
    field.height = Math.max(1, Math.min(frame.height - field.y, Math.floor(fieldInput.height)));

    var half = halfFrame(frame, field);
    var quarterImage = halveImage(half);
    // Probe order only changes how quickly the same non-negative error sum
    // reaches its existing cutoff.
    var sceneMean = [0, 0, 0], samples = 0;
    for (var at = 0; at < quarterImage.data.length; at += 291) {
      for (var channel = 0; channel < 3; channel += 1) sceneMean[channel] += quarterImage.data[at + channel];
      samples += 1;
    }
    if (samples) for (var ch = 0; ch < 3; ch += 1) sceneMean[ch] /= samples;
    var all = [];
    // boxes：候选区域（帧坐标）；为空则全游戏区扫描。
    var searchBoxes;
    if (boxes && boxes.length) {
      searchBoxes = [];
      for (var bi = 0; bi < boxes.length; bi += 1) {
        var b = boxes[bi];
        searchBoxes.push({
          x: Math.floor((b.x - field.x) / 2), y: Math.floor((b.y - field.y) / 2),
          width: Math.ceil(b.width / 2), height: Math.ceil(b.height / 2),
        });
      }
    } else searchBoxes = [{ x: 0, y: 0, width: half.width, height: half.height }];

    for (var ti = 0; ti < prepared.length; ti += 1) {
      var template = prepared[ti];
      if (!template || !template.full) continue;
      var qrgb = template.quarter.rgb;
      var contrast = function (pointAt) {
        var index = pointAt * 3;
        return (qrgb[index] - sceneMean[0]) * (qrgb[index] - sceneMean[0])
          + (qrgb[index + 1] - sceneMean[1]) * (qrgb[index + 1] - sceneMean[1])
          + (qrgb[index + 2] - sceneMean[2]) * (qrgb[index + 2] - sceneMean[2]);
      };
      var probeList = (template.quarter.probe || template.quarter.points).slice();
      probeList.sort(function (a, b2) { return contrast(b2) - contrast(a); });
      var orderedProbes = addPointCoordinates(template.quarter, probeList);

      var coarse = [];
      for (var si = 0; si < searchBoxes.length; si += 1) {
        var box = searchBoxes[si];
        // 1/4 分辨率快速筛选 → 1/2 分辨率核对（只在幸存位置附近）
        var qbox = {
          x: Math.floor(box.x / 2) - template.quarter.width, y: Math.floor(box.y / 2) - template.quarter.height,
          width: Math.ceil(box.width / 2) + template.quarter.width, height: Math.ceil(box.height / 2) + template.quarter.height,
        };
        // Thin rings/circlets have few fully opaque quarter-resolution points.
        // Skipping every second position can miss their only valid seed.
        /* ★ 2026-11-15：门槛 12 → 40。实测 27×30 的图钉类/药水类图标在 1/4 分辨率
           只剩 15~35 个前景点，它们依然会因 step=2 错过唯一有效种子（奇偶相位），
           表现为「同一个图标奇数 x 全漏、偶数 x 全中」。把门槛提到 40 让这类小图标
           直接走 step=1 全采样；点多的大图标仍走 step=2 保性能。 */
        var quarterPts = template.quarter.points.length;
        /* ★ 2026-11-15 关键修复：coarseLimit 原样传给 1/4 与 1/2 两层，
           但 1/2 层的平均误差天然比 1/4 层高（4 个像素平均后细节更多）。
           实测同一个图标落在奇数 x 时 1/2 层最佳 35.6，而 coarseLimit=34 →
           候选被整体丢弃，表现为「奇数位置全漏、偶数全中」。
           给两层分别设门槛：1/4 用 coarseLimit，1/2 用 coarseLimit*1.8（实测 35.6 可过）。 */
        var screened = coarseSearch(quarterImage, template.quarter, qbox, coarseLimit * 1.7,
          quarterPts < 40 ? 1 : 2, orderedProbes);
        /* ★ 2026-11-15 浏览器端修复：1/4 分辨率的 step=2 有奇偶相位敏感。
           实测同一个 27×30 图标在 x=301/303/305…（奇数）时全部漏检、偶数全中：
           step=2 只落了偶数格，偶数格的种子乘 2 后回到偶数半分辨率坐标，
           而真正的半分辨率位置是奇数 → 落在 half 的 ±3 核对窗口边缘之外，被丢掉。
           补一路 step=1 的精筛（只在同一 qbox 内，代价很小）覆盖面内所有相位。
           原版的这个缺陷在桌面版同样存在，但桌面版帧宽恒等于窗口宽、图标落点随机，
           用户感知为「偶尔漏」，浏览器端因为帧尺寸固定更容易稳定复现，必须修。 */
        var screenedOdd = coarseSearch(quarterImage, template.quarter, qbox, coarseLimit * 1.7, 1, orderedProbes);
        for (var so = 0; so < screenedOdd.length; so += 1) {
          var cand = screenedOdd[so], dupOdd = false;
          for (var sx2 = 0; sx2 < screened.length; sx2 += 1) {
            if (screened[sx2].x === cand.x && screened[sx2].y === cand.y) { dupOdd = true; break; }
          }
          if (!dupOdd) screened.push(cand);
        }
        screened.sort(function (a, b3) { return a.score - b3.score; });
        var seeds = [];
        for (var qi = 0; qi < screened.length; qi += 1) {
          if (seeds.length >= 40) break;
          var q = screened[qi], dup = false;
          for (var ki = 0; ki < seeds.length; ki += 1) {
            if (Math.abs(seeds[ki].x - q.x) < 4 && Math.abs(seeds[ki].y - q.y) < 4) { dup = true; break; }
          }
          if (dup) continue;
          seeds.push(q);
        }
        for (var sd = 0; sd < seeds.length; sd += 1) {
          var s0 = seeds[sd];
          /* ★ 窗口 ±3 → ±4：1/4→1/2 的坐标映射在奇偶相位下会整体偏 1px，
             6×6 窗口会把偏出去的那一列直接切掉，真位置永远拿不到候选。
             ★ 门槛 coarseLimit → coarseLimit*1.8：见上面的注释。 */
          var halfHits = coarseSearch(half, template.coarse,
            { x: s0.x * 2 - 4, y: s0.y * 2 - 4, width: 9, height: 9 }, coarseLimit * 1.8, 1);
          for (var hi = 0; hi < halfHits.length; hi += 1) coarse.push(halfHits[hi]);
        }
      }
      coarse.sort(function (a, b4) { return a.score - b4.score; });
      var kept = [];
      for (var ci = 0; ci < coarse.length; ci += 1) {
        if (kept.length >= maxPerTemplate) break;
        var c = coarse[ci], near = false;
        for (var kj = 0; kj < kept.length; kj += 1) {
          if (Math.abs(kept[kj].x - c.x) < 6 && Math.abs(kept[kj].y - c.y) < 6) { near = true; break; }
        }
        if (near) continue;
        kept.push(c);
      }
      for (var kk = 0; kk < kept.length; kk += 1) {
        var fc = kept[kk];
        var fx = field.x + fc.x * 2, fy = field.y + fc.y * 2;
        var fine = refineRenderedIcon(frame, template, fx, fy, step + 2, acceptScore);
        /* ★★ 2026-11-16 遮挡容忍（真实游戏场景实测）
           图标被怪物/角色挡掉 25% 时，位置和分数定位完全正确
           （实测坐标误差 0、visible=69%），但「平均误差」被遮挡区拉到 47，
           被 acceptScore=24 直接拒掉 → 表现为「一挡就漏」。挂机时怪物站在
           掉落物上是常态，这不是可接受的行为。

           放行条件（两条同时满足）：
             · 可见前景比例 ≥ 0.55  —— 大半个图标仍然吻合，只有局部被盖
             · 分数 ≤ acceptScore × 2.2 —— 误差升高但仍在该量级内
           误报风险实测：把阈值勉强放到 80 才出现误报，且那些候选的可见度
           都 ≤17%（纯背景巧合）；可见度 69% 只在真图标上出现，门限 0.55 余量充足。 */
        var occluded = false;
        if (!fine || fine.score > acceptScore) {
          /* ★ 遮挡放行收紧（2026-11-16 实测调参）：
             第一版只要求 visible>=0.55 就放行，结果纯色/噪点背景上大面积误报
             （单测「空背景 0 检出」直接失败，一张纯色图报出 4 个假物品）。
             原因是「可见比例」在平坦背景上天然很高：模板前景点落在同色区域时误差小，
             visible 自然满足，纯属巧合。

             收紧为三条同时满足：
               ① visible ≥ 0.60       —— 大部分前景点仍然吻合
               ② score ≤ acceptScore × 2.0
               ③ 遮挡必须落在连续区域 —— 用「最差 25% 前景点的误差远高于其余」判断，
                  真遮挡是局部一片被盖（少数点误差极大、其余极好），
                  背景巧合是整体都差（误差分布平缓）。
             实测：25% 遮挡的候选 ①②满足且 ③ 的差值为 3.4 倍；
             平坦背景的假候选差值只有 1.1~1.3 倍，能被 ③ 分开。 */
          if (fine && typeof fine.visible === 'number' && fine.visible >= 0.62
              && fine.score <= acceptScore * 2.0 && fine.occlusionRatio >= 4.5) {
            occluded = true;
          } else {
            continue;
          }
        }
        if (insideAny(excluded, fine.x, fine.y, template.full.width, template.full.height)) continue;
        all.push({
          id: template.id, x: fine.x, y: fine.y,
          width: template.full.width, height: template.full.height,
          score: fine.score, visible: fine.visible,
          occlusionRatio: fine.occlusionRatio == null ? 0 : fine.occlusionRatio,
          occluded: occluded,
        });
      }
    }
    // 同一位置多个物品竞争：保留最好者，记录与第二名的差距（区分度）。
    all.sort(function (a, b5) { return a.score - b5.score; });
    var out = [];
    for (var oi = 0; oi < all.length; oi += 1) {
      var o = all[oi], rival = null;
      for (var ri = 0; ri < out.length; ri += 1) {
        var k2 = out[ri];
        if (Math.abs(k2.x + k2.width / 2 - o.x - o.width / 2) < Math.min(k2.width, o.width) * 0.6
          && Math.abs(k2.y + k2.height / 2 - o.y - o.height / 2) < Math.min(k2.height, o.height) * 0.6) { rival = k2; break; }
      }
      if (rival) { if (rival.id !== o.id && rival.second == null) rival.second = o.score; continue; }
      out.push(o);
      if (out.length >= maxObservations) break;
    }
    for (var mi = 0; mi < out.length; mi += 1) {
      var item = out[mi];
      item.margin = item.second == null ? null : item.second - item.score;
      if (item.second === undefined) item.second = null;
    }
    return out;
  }

  return {
    MASK_ALPHA: MASK_ALPHA,
    MIN_FOREGROUND: MIN_FOREGROUND,
    prepareTemplates: prepareTemplates,
    detectFrame: detectFrame,
    detectFrameRGBA: detectFrame,

    halfFrame: halfFrame,
    coarseSearch: coarseSearch,
    fineScore: fineScore,
    resizeTemplate: resizeTemplate,
    version: '1.0.0',
  };
}));
