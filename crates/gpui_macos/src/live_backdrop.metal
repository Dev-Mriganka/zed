// Ghostex: the live backdrop styles a wallpaper-mode window can draw behind its glass
// (window_live.rs). Every style is calm, soft and slow: it is seen through a heavy tint and
// scaled up from a small texture, so fine detail and fast motion would only read as noise.
//
// Time loops: `phase` wraps at the period (two minutes at speed 1) and every style moves only
// through whole turns of `live_angle`, so the frame at the end of a loop is the frame at its
// start and the float time never loses precision. A new time term must keep that: a whole
// number of turns of `live_angle`, never raw `phase`.

#include <metal_stdlib>
using namespace metal;

struct LiveUniforms {
    float2 resolution;
    float2 view_size;
    float2 cover_origin;
    float2 cover_size;
    float phase;
    float period;
    float brightness;
    float pad;
    float4 c0;
    float4 c1;
    float4 c2;
};

struct LiveVertexOut {
    float4 position [[position]];
};

vertex LiveVertexOut live_vertex(uint vertex_id [[vertex_id]]) {
    float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    LiveVertexOut out;
    out.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

constant float LIVE_TAU = 6.28318530718;

// Per-style brightness gains (see live_finish), measured with render-posters.swift --measure so
// every style sits about 0.21 above its base at the default 60% Brightness, which still shows through
// the default glass tints (sidebar 88% / work area 81% in dark mode).
constant float LIVE_GAIN_AURORA = 2.001;
constant float LIVE_GAIN_INK = 0.894;
constant float LIVE_GAIN_DRIFT = 1.030;
constant float LIVE_GAIN_NEBULA = 2.878;
constant float LIVE_GAIN_SILK = 1.562;
constant float LIVE_GAIN_BOKEH = 3.468;
constant float LIVE_GAIN_WAVES = 1.215;
constant float LIVE_GAIN_MESH = 1.017;

static float live_angle(constant LiveUniforms &u) {
    return LIVE_TAU * u.phase / u.period;
}

// The point, in cover units: x runs with the cover's aspect, y from -0.5 at the top to 0.5.
static float2 live_point(float4 frag, constant LiveUniforms &u, thread float &aspect) {
    float2 view_point = frag.xy / u.resolution * u.view_size;
    float2 uv = (view_point - u.cover_origin) / max(u.cover_size, float2(1.0));
    aspect = u.cover_size.x / max(u.cover_size.y, 1.0);
    return float2((uv.x - 0.5) * aspect, uv.y - 0.5);
}

static float live_hash(float3 p) {
    p = fract(p * 0.3183099 + float3(0.71, 0.113, 0.419));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

static float live_noise(float3 x) {
    float3 i = floor(x);
    float3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = live_hash(i + float3(0, 0, 0));
    float n100 = live_hash(i + float3(1, 0, 0));
    float n010 = live_hash(i + float3(0, 1, 0));
    float n110 = live_hash(i + float3(1, 1, 0));
    float n001 = live_hash(i + float3(0, 0, 1));
    float n101 = live_hash(i + float3(1, 0, 1));
    float n011 = live_hash(i + float3(0, 1, 1));
    float n111 = live_hash(i + float3(1, 1, 1));
    return mix(
        mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
        mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
        f.z);
}

static float live_fbm(float3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    for (int octave = 0; octave < 4; octave++) {
        value += amplitude * live_noise(p);
        p = p * 2.03 + float3(1.7, 9.2, 3.1);
        amplitude *= 0.5;
    }
    return value;
}

// A point of the time loop: `turns` whole turns per period on a circle of `radius`, used as a
// noise offset so the field drifts and comes back to where it started.
static float3 live_orbit(float angle, float turns, float radius, float seed) {
    float a = angle * turns + seed;
    return float3(radius * cos(a), radius * sin(a), radius * 0.6 * sin(a + 1.3));
}

static float3 live_palette(float t, constant LiveUniforms &u) {
    t = clamp(t, 0.0, 1.0);
    float3 low = mix(u.c0.rgb, u.c1.rgb, smoothstep(0.0, 0.6, t));
    return mix(low, u.c2.rgb, smoothstep(0.45, 1.0, t));
}

// Brightness scales a style's distance from its deepest colour (near black in dark mode, the pale
// base in light mode). `gain` evens the styles out, so every one sits at about the same brightness
// at the same setting; a sparse style (Bokeh, Nebula) needs more than its raw colours give, so the
// scale may go past 1 and the result is clamped. Dithering hides the 8-bit steps a small texture
// shows once it is scaled up.
static float4 live_finish(float3 color, float4 frag, constant LiveUniforms &u, float gain) {
    color = u.c0.rgb + (color - u.c0.rgb) * clamp(u.brightness * gain, 0.0, 4.0);
    float dither = (live_hash(float3(frag.xy, 3.7)) - 0.5) / 255.0;
    return float4(clamp(color + dither, 0.0, 1.0), 1.0);
}

fragment float4 live_aurora(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect);
    float angle = live_angle(u);
    float3 o1 = live_orbit(angle, 1.0, 1.2, 0.0);
    float3 o2 = live_orbit(angle, 2.0, 0.8, 2.1);
    float ribbon1 = -0.08 + 1.3 * (live_fbm(float3(p.x * 0.8, 0.0, 0.0) + o1) - 0.5);
    float ribbon2 = 0.14 + 1.1 * (live_fbm(float3(p.x * 0.7, 3.0, 0.0) + o2) - 0.5);
    // A curtain hangs upward from its ribbon: it fades slowly above and quickly below.
    float d1 = p.y - ribbon1;
    float d2 = p.y - ribbon2;
    float band1 = exp(-d1 * d1 / (d1 < 0.0 ? 0.045 : 0.006));
    float band2 = exp(-d2 * d2 / (d2 < 0.0 ? 0.035 : 0.005));
    float folds = 0.7 + 0.3 * live_fbm(float3(p.x * 2.6, p.y * 0.25, 0.0) + o1 * 0.5);
    float3 color = u.c0.rgb;
    color = mix(color, u.c1.rgb, clamp(band1 * folds * 1.15, 0.0, 1.0));
    color = mix(color, u.c2.rgb, clamp(band2 * folds, 0.0, 1.0));
    return live_finish(color, in.position, u, LIVE_GAIN_AURORA);
}

fragment float4 live_ink(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect) * 1.6;
    float angle = live_angle(u);
    float3 base = float3(p, 0.0);
    float2 q = float2(
        live_fbm(base + live_orbit(angle, 1.0, 0.6, 0.0)),
        live_fbm(base + float3(5.2, 1.3, 0.0) + live_orbit(angle, 1.0, 0.6, 1.7)));
    float2 r = float2(
        live_fbm(base + float3(3.0 * q, 0.0) + float3(1.7, 9.2, 0.0) + live_orbit(angle, 2.0, 0.35, 0.4)),
        live_fbm(base + float3(3.0 * q, 0.0) + float3(8.3, 2.8, 0.0) + live_orbit(angle, 2.0, 0.35, 2.9)));
    float f = live_fbm(base + float3(3.0 * r, 0.0));
    float3 color = live_palette(f * 1.35 - 0.1, u);
    color = mix(color, u.c2.rgb, clamp(length(q) * 0.35 - 0.1, 0.0, 0.5) * 0.6);
    return live_finish(color, in.position, u, LIVE_GAIN_INK);
}

fragment float4 live_drift(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect);
    float angle = live_angle(u);
    float3 color = mix(u.c0.rgb, u.c1.rgb, 0.25);
    float3 mid = mix(u.c1.rgb, u.c2.rgb, 0.5);
    for (int i = 0; i < 5; i++) {
        float seed = float(i) * 1.618;
        float turns = (i % 2 == 0) ? 1.0 : 2.0;
        float direction = (i < 3) ? 1.0 : -1.0;
        float2 center = float2(
            0.38 * aspect * sin(angle * turns * direction + seed * 2.0),
            0.3 * cos(angle * turns + seed));
        float radius = 0.42 + 0.08 * sin(angle + seed * 3.0);
        float weight = exp(-dot(p - center, p - center) / (radius * radius));
        float3 blob = i == 0 ? u.c1.rgb : (i == 1 ? u.c2.rgb : (i == 2 ? mid : (i == 3 ? u.c1.rgb : u.c2.rgb * 0.8 + u.c0.rgb * 0.2)));
        color = mix(color, blob, clamp(weight * 0.9, 0.0, 1.0));
    }
    return live_finish(color, in.position, u, LIVE_GAIN_DRIFT);
}

fragment float4 live_nebula(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect) * 1.3;
    float angle = live_angle(u);
    float large = live_fbm(float3(p * 1.2, 0.0) + live_orbit(angle, 1.0, 0.8, 0.3));
    float small = live_fbm(float3(p * 3.0, 4.0) + live_orbit(angle, 2.0, 0.5, 1.9));
    float cloud = smoothstep(0.35, 0.85, large * 0.75 + small * 0.4);
    float glow = smoothstep(0.45, 0.95, small) * cloud;
    float3 color = mix(u.c0.rgb, u.c1.rgb, cloud);
    color = mix(color, u.c2.rgb, glow * 0.8);
    return live_finish(color, in.position, u, LIVE_GAIN_NEBULA);
}

fragment float4 live_silk(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect) * 1.2;
    float angle = live_angle(u);
    float warp = 2.0 * live_fbm(float3(p, 0.0) + live_orbit(angle, 1.0, 0.55, 0.0));
    float fold = sin((p.x + p.y * 0.6) * 4.0 + warp * 4.0 + angle * 2.0);
    float sheen = smoothstep(0.35, 1.0, fold);
    float shade = 0.5 + 0.5 * sin((p.x * 0.7 - p.y) * 2.5 + warp * 2.0 - angle);
    float3 color = mix(u.c0.rgb, u.c1.rgb, shade * 0.9);
    color = mix(color, u.c2.rgb, sheen * 0.75);
    return live_finish(color, in.position, u, LIVE_GAIN_SILK);
}

fragment float4 live_bokeh(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect);
    float angle = live_angle(u);
    float3 color = u.c0.rgb * 0.9 + u.c1.rgb * 0.1;
    for (int i = 0; i < 10; i++) {
        float seed = float(i) * 2.399;
        float turns = float(1 + i % 3);
        float2 home = float2(
            (fract(sin(seed * 12.9898) * 43758.5453) - 0.5) * aspect,
            fract(sin(seed * 78.233) * 43758.5453) - 0.5);
        float2 center = home + 0.07 * float2(cos(angle * turns + seed), sin(angle * turns + seed * 1.7));
        float radius = 0.07 + 0.09 * fract(seed * 0.37);
        float disc = exp(-dot(p - center, p - center) / (radius * radius));
        float3 light = (i % 2 == 0) ? u.c1.rgb : u.c2.rgb;
        color = mix(color, light, clamp(disc * 0.75, 0.0, 1.0));
    }
    return live_finish(color, in.position, u, LIVE_GAIN_BOKEH);
}

fragment float4 live_waves(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect);
    float angle = live_angle(u);
    float3 color = u.c0.rgb;
    for (int i = 0; i < 4; i++) {
        float layer = float(i);
        float height = -0.25 + layer * 0.17;
        float crest = height
            + 0.06 * sin(p.x * (2.0 + layer * 0.7) + angle * (1.0 + layer) + layer * 1.3)
            + 0.03 * sin(p.x * (3.7 - layer * 0.4) - angle * 2.0 + layer);
        float fill = smoothstep(crest - 0.05, crest + 0.05, p.y);
        float3 tone = live_palette(0.3 + layer * 0.22, u);
        color = mix(color, tone, fill * 0.85);
    }
    return live_finish(color, in.position, u, LIVE_GAIN_WAVES);
}

fragment float4 live_mesh(LiveVertexOut in [[stage_in]], constant LiveUniforms &u [[buffer(0)]]) {
    float aspect;
    float2 p = live_point(in.position, u, aspect);
    float angle = live_angle(u);
    float3 colors[4] = {u.c0.rgb, u.c1.rgb, u.c2.rgb, mix(u.c1.rgb, u.c2.rgb, 0.5) * 0.9 + u.c0.rgb * 0.1};
    float2 corners[4] = {
        float2(-0.35 * aspect, -0.35), float2(0.35 * aspect, -0.3),
        float2(0.3 * aspect, 0.35), float2(-0.3 * aspect, 0.3)};
    float3 color = float3(0.0);
    float total = 0.0;
    for (int i = 0; i < 4; i++) {
        float seed = float(i) * 1.9;
        float2 point = corners[i] + 0.18 * float2(cos(angle * (1.0 + float(i % 2)) + seed), sin(angle + seed * 1.3));
        float weight = 1.0 / pow(dot(p - point, p - point) + 0.02, 1.6);
        color += colors[i] * weight;
        total += weight;
    }
    return live_finish(color / total, in.position, u, LIVE_GAIN_MESH);
}

// Blends the picture that was on screen when the style or its colours changed (`from`) into the
// new one (`to`), so a change fades instead of jumping.
fragment float4 live_composite(
    LiveVertexOut in [[stage_in]],
    texture2d<float> from_texture [[texture(0)]],
    texture2d<float> to_texture [[texture(1)]],
    constant float &amount [[buffer(0)]]) {
    constexpr sampler linear_sampler(filter::linear, address::clamp_to_edge);
    float2 uv = in.position.xy / float2(to_texture.get_width(), to_texture.get_height());
    return mix(from_texture.sample(linear_sampler, uv), to_texture.sample(linear_sampler, uv), amount);
}
