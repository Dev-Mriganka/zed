// Port of gpui_macos/src/live_backdrop.metal. Keep all eight styles, gains and time loops aligned.
struct LiveUniforms {
    resolution: vec2<f32>, view_size: vec2<f32>,
    cover_origin: vec2<f32>, cover_size: vec2<f32>,
    phase: f32, period: f32, brightness: f32, style: f32,
    c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>,
};
struct GlassUniforms { current: LiveUniforms, fade: vec4<f32> };
@group(0) @binding(0) var<uniform> glass: GlassUniforms;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var previous_image: texture_2d<f32>;
@group(0) @binding(3) var current_image: texture_2d<f32>;
@vertex fn glass_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
}
const LIVE_TAU: f32 = 6.28318530718;

// Per-style brightness gains (see live_finish), measured with render-posters.swift --measure.
const LIVE_GAIN_AURORA: f32 = 1.155;
const LIVE_GAIN_INK: f32 = 0.533;
const LIVE_GAIN_DRIFT: f32 = 0.599;
const LIVE_GAIN_NEBULA: f32 = 1.860;
const LIVE_GAIN_SILK: f32 = 0.941;
const LIVE_GAIN_BOKEH: f32 = 2.041;
const LIVE_GAIN_WAVES: f32 = 0.706;
const LIVE_GAIN_MESH: f32 = 0.588;

fn live_angle(u: LiveUniforms) -> f32 {
    return LIVE_TAU * u.phase / u.period;
}

// The point, in cover units: x runs with the cover's aspect, y from -0.5 at the top to 0.5.
fn live_point(frag: vec4<f32>, u: LiveUniforms) -> vec2<f32> {
    let view_point = frag.xy / u.resolution * u.view_size;
    let uv = (view_point - u.cover_origin) / max(u.cover_size, vec2<f32>(1.0));
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    return vec2<f32>((uv.x - 0.5) * aspect, uv.y - 0.5);
}

fn live_hash(input: vec3<f32>) -> f32 {
    var p = input;
    p = fract(p * 0.3183099 + vec3<f32>(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

fn live_noise(x: vec3<f32>) -> f32 {
    var i: vec3<f32> = floor(x);
    var f: vec3<f32> = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    var n000: f32 = live_hash(i + vec3<f32>(0, 0, 0));
    var n100: f32 = live_hash(i + vec3<f32>(1, 0, 0));
    var n010: f32 = live_hash(i + vec3<f32>(0, 1, 0));
    var n110: f32 = live_hash(i + vec3<f32>(1, 1, 0));
    var n001: f32 = live_hash(i + vec3<f32>(0, 0, 1));
    var n101: f32 = live_hash(i + vec3<f32>(1, 0, 1));
    var n011: f32 = live_hash(i + vec3<f32>(0, 1, 1));
    var n111: f32 = live_hash(i + vec3<f32>(1, 1, 1));
    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z);
}

fn live_fbm(input: vec3<f32>) -> f32 {
    var p = input;
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    for (var octave: i32 = 0; octave < 4; octave += 1) {
        value += amplitude * live_noise(p);
        p = p * 2.03 + vec3<f32>(1.7, 9.2, 3.1);
        amplitude *= 0.5;
    }
    return value;
}

// A point of the time loop: `turns` whole turns per period on a circle of `radius`, used as a
// noise offset so the field drifts and comes back to where it started.
fn live_orbit(angle: f32, turns: f32, radius: f32, seed: f32) -> vec3<f32> {
    var a: f32 = angle * turns + seed;
    return vec3<f32>(radius * cos(a), radius * sin(a), radius * 0.6 * sin(a + 1.3));
}

fn live_palette(input: f32, u: LiveUniforms) -> vec3<f32> {
    let t = clamp(input, 0.0, 1.0);
    var low: vec3<f32> = mix(u.c0.rgb, u.c1.rgb, smoothstep(0.0, 0.6, t));
    return mix(low, u.c2.rgb, smoothstep(0.45, 1.0, t));
}

// Brightness dims a style toward its deepest colour (near black in dark mode, the pale base in
// light mode). `gain` evens the styles out, so every one sits at about the same brightness at the
// same setting. Dithering hides the 8-bit steps a small texture shows once it is scaled up.
fn live_finish(input: vec3<f32>, frag: vec4<f32>, u: LiveUniforms, gain: f32) -> vec4<f32> {
    var color = mix(u.c0.rgb, input, clamp(u.brightness * gain, 0.0, 1.0));
    var dither: f32 = (live_hash(vec3<f32>(frag.xy, 3.7)) - 0.5) / 255.0;
    return vec4<f32>(clamp(color + dither, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

fn live_aurora(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u);
    var angle: f32 = live_angle(u);
    var o1: vec3<f32> = live_orbit(angle, 1.0, 1.2, 0.0);
    var o2: vec3<f32> = live_orbit(angle, 2.0, 0.8, 2.1);
    var ribbon1: f32 = -0.08 + 1.3 * (live_fbm(vec3<f32>(p.x * 0.8, 0.0, 0.0) + o1) - 0.5);
    var ribbon2: f32 = 0.14 + 1.1 * (live_fbm(vec3<f32>(p.x * 0.7, 3.0, 0.0) + o2) - 0.5);
    // A curtain hangs upward from its ribbon: it fades slowly above and quickly below.
    var d1: f32 = p.y - ribbon1;
    var d2: f32 = p.y - ribbon2;
    var band1: f32 = exp(-d1 * d1 / select(0.006, 0.045, d1 < 0.0));
    var band2: f32 = exp(-d2 * d2 / select(0.005, 0.035, d2 < 0.0));
    var folds: f32 = 0.7 + 0.3 * live_fbm(vec3<f32>(p.x * 2.6, p.y * 0.25, 0.0) + o1 * 0.5);
    var color: vec3<f32> = u.c0.rgb;
    color = mix(color, u.c1.rgb, clamp(band1 * folds * 1.15, 0.0, 1.0));
    color = mix(color, u.c2.rgb, clamp(band2 * folds, 0.0, 1.0));
    return live_finish(color, frag, u, LIVE_GAIN_AURORA);
}

fn live_ink(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u) * 1.6;
    var angle: f32 = live_angle(u);
    var base: vec3<f32> = vec3<f32>(p, 0.0);
    var q: vec2<f32> = vec2<f32>(
        live_fbm(base + live_orbit(angle, 1.0, 0.6, 0.0)),
        live_fbm(base + vec3<f32>(5.2, 1.3, 0.0) + live_orbit(angle, 1.0, 0.6, 1.7)));
    var r: vec2<f32> = vec2<f32>(
        live_fbm(base + vec3<f32>(3.0 * q, 0.0) + vec3<f32>(1.7, 9.2, 0.0) + live_orbit(angle, 2.0, 0.35, 0.4)),
        live_fbm(base + vec3<f32>(3.0 * q, 0.0) + vec3<f32>(8.3, 2.8, 0.0) + live_orbit(angle, 2.0, 0.35, 2.9)));
    var f: f32 = live_fbm(base + vec3<f32>(3.0 * r, 0.0));
    var color: vec3<f32> = live_palette(f * 1.35 - 0.1, u);
    color = mix(color, u.c2.rgb, clamp(length(q) * 0.35 - 0.1, 0.0, 0.5) * 0.6);
    return live_finish(color, frag, u, LIVE_GAIN_INK);
}

fn live_drift(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u);
    var angle: f32 = live_angle(u);
    var color: vec3<f32> = mix(u.c0.rgb, u.c1.rgb, 0.25);
    var mid: vec3<f32> = mix(u.c1.rgb, u.c2.rgb, 0.5);
    for (var i: i32 = 0; i < 5; i += 1) {
        var seed: f32 = f32(i) * 1.618;
        var turns: f32 = select(2.0, 1.0, i % 2 == 0);
        var direction: f32 = select(-1.0, 1.0, i < 3);
        var center: vec2<f32> = vec2<f32>(
            0.38 * aspect * sin(angle * turns * direction + seed * 2.0),
            0.3 * cos(angle * turns + seed));
        var radius: f32 = 0.42 + 0.08 * sin(angle + seed * 3.0);
        var weight: f32 = exp(-dot(p - center, p - center) / (radius * radius));
        var blob: vec3<f32> = select(select(select(select(u.c2.rgb * 0.8 + u.c0.rgb * 0.2, u.c1.rgb, i == 3), mid, i == 2), u.c2.rgb, i == 1), u.c1.rgb, i == 0);
        color = mix(color, blob, clamp(weight * 0.9, 0.0, 1.0));
    }
    return live_finish(color, frag, u, LIVE_GAIN_DRIFT);
}

fn live_nebula(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u) * 1.3;
    var angle: f32 = live_angle(u);
    var large: f32 = live_fbm(vec3<f32>(p * 1.2, 0.0) + live_orbit(angle, 1.0, 0.8, 0.3));
    var small: f32 = live_fbm(vec3<f32>(p * 3.0, 4.0) + live_orbit(angle, 2.0, 0.5, 1.9));
    var cloud: f32 = smoothstep(0.35, 0.85, large * 0.75 + small * 0.4);
    var glow: f32 = smoothstep(0.45, 0.95, small) * cloud;
    var color: vec3<f32> = mix(u.c0.rgb, u.c1.rgb, cloud);
    color = mix(color, u.c2.rgb, glow * 0.8);
    return live_finish(color, frag, u, LIVE_GAIN_NEBULA);
}

fn live_silk(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u) * 1.2;
    var angle: f32 = live_angle(u);
    var warp: f32 = 2.0 * live_fbm(vec3<f32>(p, 0.0) + live_orbit(angle, 1.0, 0.55, 0.0));
    var fold: f32 = sin((p.x + p.y * 0.6) * 4.0 + warp * 4.0 + angle * 2.0);
    var sheen: f32 = smoothstep(0.35, 1.0, fold);
    var shade: f32 = 0.5 + 0.5 * sin((p.x * 0.7 - p.y) * 2.5 + warp * 2.0 - angle);
    var color: vec3<f32> = mix(u.c0.rgb, u.c1.rgb, shade * 0.9);
    color = mix(color, u.c2.rgb, sheen * 0.75);
    return live_finish(color, frag, u, LIVE_GAIN_SILK);
}

fn live_bokeh(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u);
    var angle: f32 = live_angle(u);
    var color: vec3<f32> = u.c0.rgb * 0.9 + u.c1.rgb * 0.1;
    for (var i: i32 = 0; i < 10; i += 1) {
        var seed: f32 = f32(i) * 2.399;
        var turns: f32 = f32(1 + i % 3);
        var home: vec2<f32> = vec2<f32>(
            (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * aspect,
            fract(sin(seed * 78.233) * 43758.5453) - 0.5);
        var center: vec2<f32> = home + 0.07 * vec2<f32>(cos(angle * turns + seed), sin(angle * turns + seed * 1.7));
        var radius: f32 = 0.07 + 0.09 * fract(seed * 0.37);
        var disc: f32 = exp(-dot(p - center, p - center) / (radius * radius));
        var light: vec3<f32> = select(u.c2.rgb, u.c1.rgb, i % 2 == 0);
        color = mix(color, light, clamp(disc * 0.75, 0.0, 1.0));
    }
    return live_finish(color, frag, u, LIVE_GAIN_BOKEH);
}

fn live_waves(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u);
    var angle: f32 = live_angle(u);
    var color: vec3<f32> = u.c0.rgb;
    for (var i: i32 = 0; i < 4; i += 1) {
        var layer: f32 = f32(i);
        var height: f32 = -0.25 + layer * 0.17;
        var crest: f32 = height
            + 0.06 * sin(p.x * (2.0 + layer * 0.7) + angle * (1.0 + layer) + layer * 1.3)
            + 0.03 * sin(p.x * (3.7 - layer * 0.4) - angle * 2.0 + layer);
        var fill: f32 = smoothstep(crest - 0.05, crest + 0.05, p.y);
        var tone: vec3<f32> = live_palette(0.3 + layer * 0.22, u);
        color = mix(color, tone, fill * 0.85);
    }
    return live_finish(color, frag, u, LIVE_GAIN_WAVES);
}

fn live_mesh(frag: vec4<f32>, u: LiveUniforms) -> vec4<f32> {
    let aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    var p: vec2<f32> = live_point(frag, u);
    var angle: f32 = live_angle(u);
    var colors = array<vec3<f32>, 4>(u.c0.rgb, u.c1.rgb, u.c2.rgb, mix(u.c1.rgb, u.c2.rgb, 0.5) * 0.9 + u.c0.rgb * 0.1);
    var corners = array<vec2<f32>, 4>(
        vec2<f32>(-0.35 * aspect, -0.35), vec2<f32>(0.35 * aspect, -0.3),
        vec2<f32>(0.3 * aspect, 0.35), vec2<f32>(-0.3 * aspect, 0.3));
    var color: vec3<f32> = vec3<f32>(0.0);
    var total: f32 = 0.0;
    for (var i: i32 = 0; i < 4; i += 1) {
        var seed: f32 = f32(i) * 1.9;
        var point: vec2<f32> = corners[i] + 0.18 * vec2<f32>(cos(angle * (1.0 + f32(i % 2)) + seed), sin(angle + seed * 1.3));
        var weight: f32 = 1.0 / pow(dot(p - point, p - point) + 0.02, 1.6);
        color += colors[i] * weight;
        total += weight;
    }
    return live_finish(color / total, frag, u, LIVE_GAIN_MESH);
}


fn sample_image(frag: vec4<f32>, u: LiveUniforms, picture: texture_2d<f32>) -> vec4<f32> {
    let size = vec2<f32>(textureDimensions(picture));
    let scale = max(u.cover_size.x / size.x, u.cover_size.y / size.y);
    let point = frag.xy / u.resolution * u.view_size - u.cover_origin;
    let uv = (point - (u.cover_size - size * scale) * 0.5) / (size * scale);
    return textureSampleLevel(picture, linear_sampler, uv, 0.0);
}
fn backdrop(frag: vec4<f32>, u: LiveUniforms, picture: texture_2d<f32>) -> vec4<f32> {
    switch u32(u.style) {
        case 1u: { return live_aurora(frag, u); }
        case 2u: { return live_ink(frag, u); }
        case 3u: { return live_drift(frag, u); }
        case 4u: { return live_nebula(frag, u); }
        case 5u: { return live_silk(frag, u); }
        case 6u: { return live_bokeh(frag, u); }
        case 7u: { return live_waves(frag, u); }
        case 8u: { return live_mesh(frag, u); }
        case 9u: { return sample_image(frag, u, picture); }
        default: { return vec4<f32>(0.0); }
    }
}
@fragment fn glass_fragment(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let current = backdrop(frag, glass.current, current_image);
    if glass.fade.x >= 1.0 { return current; }
    let previous = textureSampleLevel(previous_image, linear_sampler, frag.xy / glass.current.resolution, 0.0);
    return mix(previous, current, glass.fade.x);
}
@fragment fn glass_copy(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    return textureSampleLevel(current_image, linear_sampler, frag.xy / glass.fade.yz, 0.0);
}
