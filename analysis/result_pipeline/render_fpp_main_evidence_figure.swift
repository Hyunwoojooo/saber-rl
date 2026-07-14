#!/usr/bin/env swift
import AppKit
import Foundation

let width: CGFloat = 3000
let height: CGFloat = 1700

let outDir = URL(fileURLWithPath: "output/figures", isDirectory: true)
let pngURL = outDir.appendingPathComponent("fpp_saber_main_evidence.png")

try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let textColor = NSColor(calibratedRed: 17/255, green: 24/255, blue: 39/255, alpha: 1)
let axisColor = NSColor(calibratedRed: 31/255, green: 41/255, blue: 55/255, alpha: 1)
let gridColor = NSColor(calibratedRed: 217/255, green: 222/255, blue: 231/255, alpha: 1)
let herFill = NSColor(calibratedWhite: 0.84, alpha: 1)
let saberBase = NSColor(calibratedRed: 15/255, green: 95/255, blue: 106/255, alpha: 1)
let herStroke = NSColor(calibratedRed: 107/255, green: 114/255, blue: 128/255, alpha: 1)
let saberStroke = NSColor(calibratedRed: 6/255, green: 63/255, blue: 73/255, alpha: 1)

let backbones = ["DDPG", "TD3", "SAC"]

let finalMeans = [
    "HER": [0.940, 0.840, 0.790],
    "HER+SABER": [0.960, 0.970, 0.990],
]
let finalStd = [
    "HER": [0.042, 0.188, 0.288],
    "HER+SABER": [0.042, 0.045, 0.022],
]
let aucMeans = [
    "HER": [0.539, 0.372, 0.471],
    "HER+SABER": [0.609, 0.442, 0.583],
]
let aucStd = [
    "HER": [0.125, 0.032, 0.079],
    "HER+SABER": [0.038, 0.035, 0.030],
]
let hit90 = [
    "HER": [5, 3, 3],
    "HER+SABER": [5, 5, 5],
]

func pathLine(_ x1: CGFloat, _ y1: CGFloat, _ x2: CGFloat, _ y2: CGFloat, color: NSColor, width: CGFloat) {
    let path = NSBezierPath()
    path.lineWidth = width
    path.move(to: NSPoint(x: x1, y: y1))
    path.line(to: NSPoint(x: x2, y: y2))
    color.setStroke()
    path.stroke()
}

func drawRect(_ rect: NSRect, fill: NSColor, stroke: NSColor? = nil, strokeWidth: CGFloat = 1) {
    let path = NSBezierPath(rect: rect)
    fill.setFill()
    path.fill()
    if let stroke {
        path.lineWidth = strokeWidth
        stroke.setStroke()
        path.stroke()
    }
}

func drawHatchedRect(_ rect: NSRect) {
    drawRect(rect, fill: saberBase, stroke: saberStroke, strokeWidth: 2.2)
    NSGraphicsContext.saveGraphicsState()
    NSBezierPath(rect: rect).addClip()
    let hatch = NSColor.white.withAlphaComponent(0.55)
    hatch.setStroke()
    let spacing: CGFloat = 15
    var offset = -rect.height
    while offset < rect.width + rect.height {
        let path = NSBezierPath()
        path.lineWidth = 2.4
        path.move(to: NSPoint(x: rect.minX + offset, y: rect.maxY))
        path.line(to: NSPoint(x: rect.minX + offset + rect.height, y: rect.minY))
        path.stroke()
        offset += spacing
    }
    NSGraphicsContext.restoreGraphicsState()
}

func font(_ size: CGFloat, weight: NSFont.Weight = .regular) -> NSFont {
    NSFont.systemFont(ofSize: size, weight: weight)
}

func drawText(
    _ value: String,
    x: CGFloat,
    y: CGFloat,
    size: CGFloat,
    weight: NSFont.Weight = .regular,
    color: NSColor = textColor,
    align: NSTextAlignment = .left,
    rotateDegrees: CGFloat? = nil
) {
    let attrs: [NSAttributedString.Key: Any] = [
        .font: font(size, weight: weight),
        .foregroundColor: color,
    ]
    let ns = NSString(string: value)
    let textSize = ns.size(withAttributes: attrs)
    var drawX = x
    let drawY = y
    if align == .center {
        drawX -= textSize.width / 2
    } else if align == .right {
        drawX -= textSize.width
    }

    if let degrees = rotateDegrees {
        NSGraphicsContext.saveGraphicsState()
        let transform = NSAffineTransform()
        transform.translateX(by: x, yBy: y)
        transform.rotate(byDegrees: degrees)
        transform.concat()
        let rotatedX = align == .center ? -textSize.width / 2 : (align == .right ? -textSize.width : 0)
        ns.draw(in: NSRect(x: rotatedX, y: -textSize.height / 2, width: textSize.width + 4, height: textSize.height + 4), withAttributes: attrs)
        NSGraphicsContext.restoreGraphicsState()
    } else {
        ns.draw(in: NSRect(x: drawX, y: drawY, width: textSize.width + 8, height: textSize.height + 8), withAttributes: attrs)
    }
}

func yMap(_ value: Double, yMin: Double, yMax: Double, plotTop: CGFloat, plotBottom: CGFloat) -> CGFloat {
    let clipped = min(max(value, yMin), yMax)
    return plotBottom - CGFloat((clipped - yMin) / (yMax - yMin)) * (plotBottom - plotTop)
}

struct PanelSpec {
    let letter: String
    let title: String
    let yLabel: String
    let x: CGFloat
    let y: CGFloat
    let width: CGFloat
    let plotTop: CGFloat
    let plotBottom: CGFloat
    let yMin: Double
    let yMax: Double
    let ticks: [Double]
}

func drawPanelAxes(_ spec: PanelSpec) -> (plotLeft: CGFloat, plotRight: CGFloat, groupWidth: CGFloat) {
    let plotLeft = spec.x + 138
    let plotRight = spec.x + spec.width - 34
    let groupWidth = (plotRight - plotLeft) / 3

    drawText("(\(spec.letter)) \(spec.title)", x: spec.x + 4, y: spec.y, size: 58, weight: .bold)
    drawText(
        spec.yLabel,
        x: spec.x + 34,
        y: (spec.plotTop + spec.plotBottom) / 2,
        size: 54,
        weight: .semibold,
        align: .center,
        rotateDegrees: -90
    )

    for tick in spec.ticks {
        let y = yMap(tick, yMin: spec.yMin, yMax: spec.yMax, plotTop: spec.plotTop, plotBottom: spec.plotBottom)
        pathLine(plotLeft, y, plotRight, y, color: gridColor, width: 2)
        let label: String
        if spec.yMax == 5 {
            label = "\(Int(tick))"
        } else {
            label = tick == 0 ? "0.00" : String(format: "%.2f", tick)
        }
        drawText(label, x: plotLeft - 20, y: y - 26, size: 46, color: textColor, align: .right)
    }

    pathLine(plotLeft, spec.plotBottom, plotRight, spec.plotBottom, color: axisColor, width: 2.4)
    pathLine(plotLeft, spec.plotTop, plotLeft, spec.plotBottom, color: axisColor, width: 2.4)

    for (idx, name) in backbones.enumerated() {
        let cx = plotLeft + groupWidth * (CGFloat(idx) + 0.5)
        drawText(name, x: cx, y: spec.plotBottom + 36, size: 54, weight: .semibold, align: .center)
    }

    return (plotLeft, plotRight, groupWidth)
}

func drawBars(
    spec: PanelSpec,
    means: [String: [Double]],
    stds: [String: [Double]],
    annotations: [String]? = nil
) {
    let axes = drawPanelAxes(spec)
    let barWidth: CGFloat = 66
    let barGap: CGFloat = 16
    let series = ["HER", "HER+SABER"]

    for (sIdx, seriesName) in series.enumerated() {
        for idx in 0..<3 {
            let cx = axes.plotLeft + axes.groupWidth * (CGFloat(idx) + 0.5)
            let x = cx + (sIdx == 0 ? -barWidth - barGap / 2 : barGap / 2)
            let mean = means[seriesName]![idx]
            let std = stds[seriesName]![idx]
            let y = yMap(mean, yMin: spec.yMin, yMax: spec.yMax, plotTop: spec.plotTop, plotBottom: spec.plotBottom)
            let rect = NSRect(x: x, y: y, width: barWidth, height: spec.plotBottom - y)
            if seriesName == "HER" {
                drawRect(rect, fill: herFill, stroke: herStroke, strokeWidth: 2.2)
            } else {
                drawHatchedRect(rect)
            }

            let errTop = yMap(min(mean + std, spec.yMax), yMin: spec.yMin, yMax: spec.yMax, plotTop: spec.plotTop, plotBottom: spec.plotBottom)
            let errBottom = yMap(max(mean - std, spec.yMin), yMin: spec.yMin, yMax: spec.yMax, plotTop: spec.plotTop, plotBottom: spec.plotBottom)
            let errX = x + barWidth / 2
            pathLine(errX, errTop, errX, errBottom, color: textColor, width: 3)
            pathLine(errX - 16, errTop, errX + 16, errTop, color: textColor, width: 3)
            pathLine(errX - 16, errBottom, errX + 16, errBottom, color: textColor, width: 3)
        }
    }

    if let annotations {
        for (idx, label) in annotations.enumerated() {
            let cx = axes.plotLeft + axes.groupWidth * (CGFloat(idx) + 0.5)
            for (lineIdx, line) in label.components(separatedBy: "\n").enumerated() {
                drawText(line, x: cx, y: spec.plotTop + 10 + CGFloat(lineIdx) * 34, size: 32, color: NSColor(calibratedRed: 55/255, green: 65/255, blue: 81/255, alpha: 1), align: .center)
            }
        }
    }
}

func drawHitPanel(spec: PanelSpec) {
    let axes = drawPanelAxes(spec)
    let barWidth: CGFloat = 66
    let barGap: CGFloat = 16
    let series = ["HER", "HER+SABER"]

    for (sIdx, seriesName) in series.enumerated() {
        for idx in 0..<3 {
            let cx = axes.plotLeft + axes.groupWidth * (CGFloat(idx) + 0.5)
            let x = cx + (sIdx == 0 ? -barWidth - barGap / 2 : barGap / 2)
            let value = hit90[seriesName]![idx]
            let y = yMap(Double(value), yMin: spec.yMin, yMax: spec.yMax, plotTop: spec.plotTop, plotBottom: spec.plotBottom)
            let rect = NSRect(x: x, y: y, width: barWidth, height: spec.plotBottom - y)
            if seriesName == "HER" {
                drawRect(rect, fill: herFill, stroke: herStroke, strokeWidth: 2.2)
            } else {
                drawHatchedRect(rect)
            }
            let labelColor = seriesName == "HER" ? textColor : NSColor.white
            drawText("\(value)/5", x: x + barWidth / 2, y: max(spec.plotTop + 14, y - 50), size: 44, weight: .bold, color: labelColor, align: .center)
        }
    }
}

let image = NSImage(size: NSSize(width: width, height: height))
image.lockFocusFlipped(true)

NSColor.white.setFill()
NSRect(x: 0, y: 0, width: width, height: height).fill()

let panelY: CGFloat = 118
let plotTop: CGFloat = 282
let plotBottom: CGFloat = 1328
let marginX: CGFloat = 92
let gap: CGFloat = 86
let panelWidth = (width - 2 * marginX - 2 * gap) / 3

drawBars(
    spec: PanelSpec(letter: "a", title: "Final success", yLabel: "Final success", x: marginX, y: panelY, width: panelWidth, plotTop: plotTop, plotBottom: plotBottom, yMin: 0, yMax: 1.05, ticks: [0, 0.25, 0.50, 0.75, 1.00]),
    means: finalMeans,
    stds: finalStd
)

drawBars(
    spec: PanelSpec(letter: "b", title: "Success AUC", yLabel: "Success AUC", x: marginX + panelWidth + gap, y: panelY, width: panelWidth, plotTop: plotTop, plotBottom: plotBottom, yMin: 0, yMax: 0.75, ticks: [0, 0.25, 0.50, 0.75]),
    means: aucMeans,
    stds: aucStd
)

drawHitPanel(
    spec: PanelSpec(letter: "c", title: "Hit90 reached seeds", yLabel: "Hit90 reached seeds", x: marginX + 2 * (panelWidth + gap), y: panelY, width: panelWidth, plotTop: plotTop, plotBottom: plotBottom, yMin: 0, yMax: 5, ticks: [0, 1, 2, 3, 4, 5])
)

let legendX: CGFloat = width - 588
let legendY: CGFloat = 42
drawRect(NSRect(x: legendX, y: legendY, width: 54, height: 34), fill: herFill, stroke: herStroke, strokeWidth: 2)
drawText("HER", x: legendX + 70, y: legendY - 7, size: 46)
drawHatchedRect(NSRect(x: legendX + 220, y: legendY, width: 54, height: 34))
drawText("HER+SABER", x: legendX + 290, y: legendY - 7, size: 46)

let stripY: CGFloat = 1586
pathLine(90, stripY, width - 90, stripY, color: NSColor(calibratedRed: 184/255, green: 192/255, blue: 204/255, alpha: 1), width: 2)
drawText(
    "Matched one-update-per-step HER backbones; error bars show seed standard deviation.",
    x: width / 2,
    y: stripY + 31,
    size: 46,
    color: axisColor,
    align: .center
)

image.unlockFocus()

guard let tiff = image.tiffRepresentation,
      let rep = NSBitmapImageRep(data: tiff),
      let png = rep.representation(using: .png, properties: [:]) else {
    fatalError("Failed to encode PNG")
}

try png.write(to: pngURL)
print(pngURL.path)
