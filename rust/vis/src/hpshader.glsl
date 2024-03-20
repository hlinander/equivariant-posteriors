#version 450
#ifdef VERTEX
#define VERTEXARG out
#endif
#ifdef FRAGMENT
#define VERTEXARG in
#endif

#define PI 3.14150265358

layout(location = 0) VERTEXARG vec2 uv;

#ifdef FRAGMENT
layout(set = 0, binding = 0) uniform Uniforms{
    float angle1;
    float angle2;
    float min;
    float max;
    int nside;
};
layout(std430, set = 0, binding = 1) readonly buffer HPBuffers{
    float hp_data[];
};
#endif

#ifdef VERTEX
// perfect never change
void main() {
    float x = float((gl_VertexIndex & 1) << 2) - 1.0f;
    float y = float((gl_VertexIndex & 2) - 1.0f);
    gl_Position = vec4(x, y, 0.0, 1.0);
    uv.x = ((gl_VertexIndex & 1) << 2) * 0.5;
    uv.y = (gl_VertexIndex & 2) * 0.5;
}
#endif
float luminance(vec3 rgb) {
    return dot(rgb, vec3(0.2125, 0.7154, 0.0721));
}


vec3 plasma_quintic( float x )
{
	x = clamp( x, 0.0, 1.0 );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	return vec3(
		dot( x1.xyzw, vec4( +0.063861086, +1.992659096, -1.023901152, -0.490832805 ) ) + dot( x2.xy, vec2( +1.308442123, -0.914547012 ) ),
		dot( x1.xyzw, vec4( +0.049718590, -0.791144343, +2.892305078, +0.811726816 ) ) + dot( x2.xy, vec2( -4.686502417, +2.717794514 ) ),
		dot( x1.xyzw, vec4( +0.513275779, +1.580255060, -5.164414457, +4.559573646 ) ) + dot( x2.xy, vec2( -1.916810682, +0.570638854 ) ) );
}


vec3 magma_quintic( float x )
{
	x = clamp( x, 0.0, 1.0 );
	vec4 x1 = vec4( 1.0, x, x * x, x * x * x ); // 1 x x2 x3
	vec4 x2 = x1 * x1.w * x; // x4 x5 x6 x7
	vec3 rgb = vec3(
		dot( x1.xyzw, vec4( -0.023226960, +1.087154378, -0.109964741, +6.333665763 ) ) + dot( x2.xy, vec2( -11.640596589, +5.337625354 ) ),
		dot( x1.xyzw, vec4( +0.010680993, +0.176613780, +1.638227448, -6.743522237 ) ) + dot( x2.xy, vec2( +11.426396979, -5.523236379 ) ),
		dot( x1.xyzw, vec4( -0.008260782, +2.244286052, +3.005587601, -24.279769818 ) ) + dot( x2.xy, vec2( +32.484310068, -12.688259703 ) ) );
    return clamp(rgb, 0.0, 1.0);
}


float sphIntersect( in vec3 ro, in vec3 rd, in vec4 sph )
{
	vec3 oc = ro - sph.xyz;
	float b = dot( oc, rd );
	float c = dot( oc, oc ) - sph.w*sph.w;
	float h = b*b - c;
	if( h<0.0 ) return -1.0;
	return -b - sqrt( h );
}

#ifdef FRAGMENT

const int x2pix[] = int[](0, 1, 4, 5, 16, 17, 20, 21, 64, 65, 68, 69, 80, 81, 84, 85, 256, 257, 260, 261, 272, 273, 276, 277, 320, 321, 324, 325, 336, 337, 340, 341, 1024, 1025, 1028, 1029, 1040, 1041, 1044, 1045, 1088, 1089, 1092, 1093, 1104, 1105, 1108, 1109, 1280, 1281, 1284, 1285, 1296, 1297, 1300, 1301, 1344, 1345, 1348, 1349, 1360, 1361, 1364, 1365, 4096, 4097, 4100, 4101, 4112, 4113, 4116, 4117, 4160, 4161, 4164, 4165, 4176, 4177, 4180, 4181, 4352, 4353, 4356, 4357, 4368, 4369, 4372, 4373, 4416, 4417, 4420, 4421, 4432, 4433, 4436, 4437, 5120, 5121, 5124, 5125, 5136, 5137, 5140, 5141, 5184, 5185, 5188, 5189, 5200, 5201, 5204, 5205, 5376, 5377, 5380, 5381, 5392, 5393, 5396, 5397, 5440, 5441, 5444, 5445, 5456, 5457, 5460, 5461);
const int y2pix[] = int[](0, 2, 8, 10, 32, 34, 40, 42, 128, 130, 136, 138, 160, 162, 168, 170, 512, 514, 520, 522, 544, 546, 552, 554, 640, 642, 648, 650, 672, 674, 680, 682, 2048, 2050, 2056, 2058, 2080, 2082, 2088, 2090, 2176, 2178, 2184, 2186, 2208, 2210, 2216, 2218, 2560, 2562, 2568, 2570, 2592, 2594, 2600, 2602, 2688, 2690, 2696, 2698, 2720, 2722, 2728, 2730, 8192, 8194, 8200, 8202, 8224, 8226, 8232, 8234, 8320, 8322, 8328, 8330, 8352, 8354, 8360, 8362, 8704, 8706, 8712, 8714, 8736, 8738, 8744, 8746, 8832, 8834, 8840, 8842, 8864, 8866, 8872, 8874, 10240, 10242, 10248, 10250, 10272, 10274, 10280, 10282, 10368, 10370, 10376, 10378, 10400, 10402, 10408, 10410, 10752, 10754, 10760, 10762, 10784, 10786, 10792, 10794, 10880, 10882, 10888, 10890, 10912, 10914, 10920, 10922);
// const int ns_max = 8192;
const int ns_max = 4096;
const float piover2 = PI * 0.5;
const float pi = PI;
const float twopi = 2.0 * PI;

int vec2pix_nest(int nside, vec3 vec) {
    float z, za, z0, tt, tp, tmp, phi;
    int face_num, jp, jm, ifp, ifm;
    int ix, iy, ix_low, ix_hi, iy_low, iy_hi, ipf, ntt;

    if (nside < 1 || nside > ns_max) {
        // Error handling, might just return -1 or handle as best as possible
        return -1;
    }

    z = vec.z / length(vec);
    phi = 0.0;
    if (vec.x != 0.0 || vec.y != 0.0) {
        phi = atan(vec.y, vec.x); // atan in GLSL handles two arguments and returns in ]-pi, pi]
        if (phi < 0.0) phi += twopi; // Normalize to [0, 2pi[
    }

    za = abs(z);
    z0 = 2.0 / 3.0;
    tt = phi / piover2;

    if (za <= z0) {
        jp = int(floor(ns_max * (0.5 + tt - z * 0.75)));
        jm = int(floor(ns_max * (0.5 + tt + z * 0.75)));

        ifp = jp / ns_max;
        ifm = jm / ns_max;

        face_num = ifp == ifm ? int(mod(float(ifp), 4.0)) + 4 : ifp < ifm ? int(mod(float(ifp), 4.0)) : int(mod(float(ifm), 4.0)) + 8;

        ix = int(mod(float(jm), float(ns_max)));
        iy = ns_max - int(mod(float(jp), float(ns_max))) - 1;
    } else {
        ntt = int(floor(tt));
        if (ntt >= 4) ntt = 3;
        tp = tt - float(ntt);
        tmp = sqrt(3.0 * (1.0 - za));

        jp = int(floor(ns_max * tp * tmp));
        jm = int(floor(ns_max * (1.0 - tp) * tmp));
        jp = jp < ns_max - 1 ? jp : ns_max - 1;
        jm = jm < ns_max - 1 ? jm : ns_max - 1;

        if (z >= 0.0) {
            face_num = ntt;
            ix = ns_max - jm - 1;
            iy = ns_max - jp - 1;
        } else {
            face_num = ntt + 8;
            ix = jp;
            iy = jm;
        }
    }

    ix_low = int(mod(float(ix), 128.0));
    ix_hi = ix / 128;
    iy_low = int(mod(float(iy), 128.0));
    iy_hi = iy / 128;

    ipf = (x2pix[ix_hi] + y2pix[iy_hi]) * 128 * 128 + (x2pix[ix_low] + y2pix[iy_low]);
    ipf = int(float(ipf) / pow(float(ns_max / nside), 2.0)); // Adjust for resolution

    int ipix = ipf + face_num * int(pow(float(nside), 2.0));
    return ipix;
}

vec3 rotateVector(vec3 v, vec3 axis, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    float oneminusc = 1.0 - c;
    float x = axis.x, y = axis.y, z = axis.z;

    mat3 rot = mat3(
        x*x*oneminusc + c, x*y*oneminusc - z*s, x*z*oneminusc + y*s,
        y*x*oneminusc + z*s, y*y*oneminusc + c, y*z*oneminusc - x*s,
        z*x*oneminusc - y*s, z*y*oneminusc + x*s, z*z*oneminusc + c
    );

    return rot * v;
}

layout(location = 0) out vec4 color;
void main() {
    // float d = uv.length();
    // color = vec4(uv.x, uv.y, 0.0, 1.0);
    vec2 uv_center = uv - vec2(0.5, 0.5);
    uv_center.y = -uv_center.y;
    vec3 ro = vec3(0, 0, -2);
    vec3 rd = normalize(vec3(uv_center, 2));

    ro = rotateVector(ro, vec3(1, 0, 0), PI/2.0 + angle2);
    rd = rotateVector(rd, vec3(1, 0, 0), PI/2.0 + angle2);

    ro = rotateVector(ro, vec3(0, 0, 1), -PI/2.0 - angle1);
    rd = rotateVector(rd, vec3(0, 0, 1), -PI/2.0 - angle1);

    vec4 sph = vec4(0, 0, 0, 0.4);
    float t = sphIntersect(ro, rd, sph);
    vec3 col = vec3(0);
    if (t > 0.0) {
        vec3 pos = ro + t*rd;
        vec3 nor = normalize( pos - sph.xyz );
        // nor = rotateVector(nor, vec3(0, 1, 0), -angle1);
        // nor = rotateVector(nor, vec3(1, 0, 0), 3.14159 / 2.0 + angle2);
        int hp = vec2pix_nest(nside, nor);
        // int npix = 12 * 64 * 64;
        col = plasma_quintic((hp_data[hp] - min) / (max - min));
        // col = plasma_quintic(float(hp) / float(npix));
        // col = vec3(1.2);
        // col *= 0.6+0.4*nor.y;
    }
    color = vec4(col, 1.0);
}
#endif

