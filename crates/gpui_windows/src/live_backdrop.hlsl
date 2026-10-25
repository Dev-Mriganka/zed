// Ghostex: the Windows port of gpui_macos/src/live_backdrop.metal, the live backdrop styles a
// wallpaper-mode window draws behind its glass (directx_backdrop.rs), plus the picture and
// cross-fade passes the Windows backdrop shares with them. Every style must stay identical to the
// Metal one, line for line, so both platforms draw the same frame at the same phase.
//
// Time loops: `phase` wraps at the period (two minutes at speed 1) and every style moves only
// through whole turns of `live_angle`, so the frame at the end of a loop is the frame at its
// start and the float time never loses precision. A new time term must keep that: a whole number
// of turns of `live_angle`, never raw `phase`.

cbuffer LiveUniforms : register(b0) {
    float2 resolution;
    float2 view_size;
    float2 cover_origin;
    float2 cover_size;
    float phase;
    float period;
    float brightness;
    float amount;
    float4 c0;
    float4 c1;
    float4 c2;
};

Texture2D from_texture : register(t0);
Texture2D to_texture : register(t1);
SamplerState linear_sampler : register(s0);

float4 live_vertex(uint vertex_id : SV_VertexID) : SV_Position {
    float2 corner = float2((vertex_id << 1) & 2, vertex_id & 2);
    return float4(corner * 2.0 - 1.0, 0.0, 1.0);
}

static const float LIVE_TAU = 6.28318530718;

// Per-style brightness gains (see live_finish), measured with render-posters.swift --measure.
static const float LIVE_GAIN_AURORA = 1.155;
static const float LIVE_GAIN_INK = 0.533;
static const float LIVE_GAIN_DRIFT = 0.599;
static const float LIVE_GAIN_NEBULA = 1.860;
static const float LIVE_GAIN_SILK = 0.941;
static const float LIVE_GAIN_BOKEH = 2.041;
static const float LIVE_GAIN_WAVES = 0.706;
static const float LIVE_GAIN_MESH = 0.588;

float live_angle() {
    return LIVE_TAU * phase / period;
}

// The point, in cover units: x runs with the cover's aspect, y from -0.5 at the top to 0.5.
float2 live_point(float4 frag, out float aspect) {
    float2 view_point = frag.xy / resolution * view_size;
    float2 uv = (view_point - cover_origin) / max(cover_size, float2(1.0, 1.0));
    aspect = cover_size.x / max(cover_size.y, 1.0);
    return float2((uv.x - 0.5) * aspect, uv.y - 0.5);
}

float live_hash(float3 p) {
    p = frac(p * 0.3183099 + float3(0.71, 0.113, 0.419));
    p *= 17.0;
    return frac(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float live_noise(float3 x) {
    float3 i = floor(x);
    float3 f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = live_hash(i + float3(0, 0, 0));
    float n100 = live_hash(i + float3(1, 0, 0));
    float n010 = live_hash(i + float3(0, 1, 0));
    float n110 = live_hash(i + float3(1, 1, 0));
    float n001 = live_hash(i + float3(0, 0, 1));
    float n101 = live_hash(i + float3(1, 0, 1));
    float n011 = live_hash(i + float3(0, 1, 1));
    float n111 = live_hash(i + float3(1, 1, 1));
    return lerp(
        lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
        lerp(lerp(n001, n101, f.x), lerp(n011, n111, f.x), f.y),
        f.z);
}

float live_fbm(float3 p) {
    float value = 0.0;
    float amplitude = 0.5;
    [unroll] for (int octave = 0; octave < 4; octave++) {
        value += amplitude * live_noise(p);
        p = p * 2.03 + float3(1.7, 9.2, 3.1);
        amplitude *= 0.5;
    }
    return value;
}

// A point of the time loop: `turns` whole turns per period on a circle of `radius`, used as a
// noise offset so the field drifts and comes back to where it started.
float3 live_orbit(float angle, float turns, float radius, float seed) {
    float a = angle * turns + seed;
    return float3(radius * cos(a), radius * sin(a), radius * 0.6 * sin(a + 1.3));
}

float3 live_palette(float t) {
    t = clamp(t, 0.0, 1.0);
    float3 low = lerp(c0.rgb, c1.rgb, smoothstep(0.0, 0.6, t));
    return lerp(low, c2.rgb, smoothstep(0.45, 1.0, t));
}

// Brightness dims a style toward its deepest colour (near black in dark mode, the pale base in
// light mode). `gain` evens the styles out, so every one sits at about the same brightness at the
// same setting. Dithering hides the 8-bit steps a small texture shows once it is scaled up.
float4 live_finish(float3 color, float4 frag, float gain) {
    color = lerp(c0.rgb, color, clamp(brightness * gain, 0.0, 1.0));
    float dither = (live_hash(float3(frag.xy, 3.7)) - 0.5) / 255.0;
    return float4(clamp(color + dither, 0.0, 1.0), 1.0);
}

float4 live_aurora(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect);
    float angle = live_angle();
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
    float3 color = c0.rgb;
    color = lerp(color, c1.rgb, clamp(band1 * folds * 1.15, 0.0, 1.0));
    color = lerp(color, c2.rgb, clamp(band2 * folds, 0.0, 1.0));
    return live_finish(color, position, LIVE_GAIN_AURORA);
}

float4 live_ink(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect) * 1.6;
    float angle = live_angle();
    float3 base = float3(p, 0.0);
    float2 q = float2(
        live_fbm(base + live_orbit(angle, 1.0, 0.6, 0.0)),
        live_fbm(base + float3(5.2, 1.3, 0.0) + live_orbit(angle, 1.0, 0.6, 1.7)));
    float2 r = float2(
        live_fbm(base + float3(3.0 * q, 0.0) + float3(1.7, 9.2, 0.0) + live_orbit(angle, 2.0, 0.35, 0.4)),
        live_fbm(base + float3(3.0 * q, 0.0) + float3(8.3, 2.8, 0.0) + live_orbit(angle, 2.0, 0.35, 2.9)));
    float f = live_fbm(base + float3(3.0 * r, 0.0));
    float3 color = live_palette(f * 1.35 - 0.1);
    color = lerp(color, c2.rgb, clamp(length(q) * 0.35 - 0.1, 0.0, 0.5) * 0.6);
    return live_finish(color, position, LIVE_GAIN_INK);
}

float4 live_drift(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect);
    float angle = live_angle();
    float3 color = lerp(c0.rgb, c1.rgb, 0.25);
    float3 mid = lerp(c1.rgb, c2.rgb, 0.5);
    [unroll] for (int i = 0; i < 5; i++) {
        float seed = float(i) * 1.618;
        float turns = (i % 2 == 0) ? 1.0 : 2.0;
        float direction = (i < 3) ? 1.0 : -1.0;
        float2 center = float2(
            0.38 * aspect * sin(angle * turns * direction + seed * 2.0),
            0.3 * cos(angle * turns + seed));
        float radius = 0.42 + 0.08 * sin(angle + seed * 3.0);
        float weight = exp(-dot(p - center, p - center) / (radius * radius));
        float3 blob = i == 0 ? c1.rgb : (i == 1 ? c2.rgb : (i == 2 ? mid : (i == 3 ? c1.rgb : c2.rgb * 0.8 + c0.rgb * 0.2)));
        color = lerp(color, blob, clamp(weight * 0.9, 0.0, 1.0));
    }
    return live_finish(color, position, LIVE_GAIN_DRIFT);
}

float4 live_nebula(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect) * 1.3;
    float angle = live_angle();
    float large = live_fbm(float3(p * 1.2, 0.0) + live_orbit(angle, 1.0, 0.8, 0.3));
    float small = live_fbm(float3(p * 3.0, 4.0) + live_orbit(angle, 2.0, 0.5, 1.9));
    float cloud = smoothstep(0.35, 0.85, large * 0.75 + small * 0.4);
    float glow = smoothstep(0.45, 0.95, small) * cloud;
    float3 color = lerp(c0.rgb, c1.rgb, cloud);
    color = lerp(color, c2.rgb, glow * 0.8);
    return live_finish(color, position, LIVE_GAIN_NEBULA);
}

float4 live_silk(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect) * 1.2;
    float angle = live_angle();
    float warp = 2.0 * live_fbm(float3(p, 0.0) + live_orbit(angle, 1.0, 0.55, 0.0));
    float fold = sin((p.x + p.y * 0.6) * 4.0 + warp * 4.0 + angle * 2.0);
    float sheen = smoothstep(0.35, 1.0, fold);
    float shade = 0.5 + 0.5 * sin((p.x * 0.7 - p.y) * 2.5 + warp * 2.0 - angle);
    float3 color = lerp(c0.rgb, c1.rgb, shade * 0.9);
    color = lerp(color, c2.rgb, sheen * 0.75);
    return live_finish(color, position, LIVE_GAIN_SILK);
}

float4 live_bokeh(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect);
    float angle = live_angle();
    float3 color = c0.rgb * 0.9 + c1.rgb * 0.1;
    [unroll] for (int i = 0; i < 10; i++) {
        float seed = float(i) * 2.399;
        float turns = float(1 + i % 3);
        float2 home = float2(
            (frac(sin(seed * 12.9898) * 43758.5453) - 0.5) * aspect,
            frac(sin(seed * 78.233) * 43758.5453) - 0.5);
        float2 center = home + 0.07 * float2(cos(angle * turns + seed), sin(angle * turns + seed * 1.7));
        float radius = 0.07 + 0.09 * frac(seed * 0.37);
        float disc = exp(-dot(p - center, p - center) / (radius * radius));
        float3 light = (i % 2 == 0) ? c1.rgb : c2.rgb;
        color = lerp(color, light, clamp(disc * 0.75, 0.0, 1.0));
    }
    return live_finish(color, position, LIVE_GAIN_BOKEH);
}

float4 live_waves(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect);
    float angle = live_angle();
    float3 color = c0.rgb;
    [unroll] for (int i = 0; i < 4; i++) {
        float layer = float(i);
        float height = -0.25 + layer * 0.17;
        float crest = height
            + 0.06 * sin(p.x * (2.0 + layer * 0.7) + angle * (1.0 + layer) + layer * 1.3)
            + 0.03 * sin(p.x * (3.7 - layer * 0.4) - angle * 2.0 + layer);
        float fill = smoothstep(crest - 0.05, crest + 0.05, p.y);
        float3 tone = live_palette(0.3 + layer * 0.22);
        color = lerp(color, tone, fill * 0.85);
    }
    return live_finish(color, position, LIVE_GAIN_WAVES);
}

float4 live_mesh(float4 position : SV_Position) : SV_Target {
    float aspect;
    float2 p = live_point(position, aspect);
    float angle = live_angle();
    float3 colors[4] = {c0.rgb, c1.rgb, c2.rgb, lerp(c1.rgb, c2.rgb, 0.5) * 0.9 + c0.rgb * 0.1};
    float2 corners[4] = {
        float2(-0.35 * aspect, -0.35), float2(0.35 * aspect, -0.3),
        float2(0.3 * aspect, 0.35), float2(-0.3 * aspect, 0.3)};
    float3 color = float3(0.0, 0.0, 0.0);
    float total = 0.0;
    [unroll] for (int i = 0; i < 4; i++) {
        float seed = float(i) * 1.9;
        float2 target_point = corners[i] + 0.18 * float2(cos(angle * (1.0 + float(i % 2)) + seed), sin(angle + seed * 1.3));
        float weight = 1.0 / pow(dot(p - target_point, p - target_point) + 0.02, 1.6);
        color += colors[i] * weight;
        total += weight;
    }
    return live_finish(color / total, position, LIVE_GAIN_MESH);
}

// Windows only: the blurred picture (desktop wallpaper or the chosen image), laid out so it fills
// the rectangle `cover_origin`/`cover_size`, in the backdrop's own pixels.
float4 backdrop_picture(float4 position : SV_Position) : SV_Target {
    float2 uv = (position.xy - cover_origin) / max(cover_size, float2(1.0, 1.0));
    return float4(to_texture.Sample(linear_sampler, saturate(uv)).rgb, 1.0);
}

// Blends the picture that was on screen when the style, its colours or the picture changed
// (`from`) into the new one (`to`), so a change fades instead of jumping.
float4 backdrop_composite(float4 position : SV_Position) : SV_Target {
    float width;
    float height;
    to_texture.GetDimensions(width, height);
    float2 uv = position.xy / float2(width, height);
    return lerp(from_texture.Sample(linear_sampler, uv), to_texture.Sample(linear_sampler, uv), amount);
}
