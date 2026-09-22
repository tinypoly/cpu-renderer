import { describe, expect, it } from "vitest";
import { CpuShader, matrixValue } from "./glsl.js";
import { ShaderChunk } from "three";

const context = { texture: () => [1, .5, .25, 1] };
const run = (source: string) => new CpuShader(source).run({}, context)!;

describe("CPU GLSL compatibility", () => {
  it("executes Three's actual common include with macros, light structs and overloads", () => {
    const shader = new CpuShader(`#include <common>
void main() {
  IncidentLight light=IncidentLight(vec3(0.5,1.0,2.0),vec3(0.0,1.0,0.0),true);
  gl_FragColor=vec4(saturate(pow2(light.color)),pow2(0.5));
}`, {}, ShaderChunk);

    expect(shader.run({}, context)!.gl_FragColor).toEqual([.25, 1, 1, .25]);
  });
  it("expands nested and multiline function macros without rewriting numeric tokens", () => {
    const shader = new CpuShader(`
#define e 99
#define SCALE(x, y) ((x) * (y))
#define APPLY SCALE
#define DOUBLE(x) APPLY(x, 2.0)
#define EMPTY()
void main() { EMPTY() gl_FragColor = vec4(DOUBLE(
  SCALE(vec3(1e-2, 2e-2, 3e-2), 2.0)
), 1.0); }`);

    expect(shader.run({}, context)!.gl_FragColor).toEqual([.04, .08, .12, 1]);
  });
  it("shares include defines and undefines, respects guards and ignores directives inside comments", () => {
    const shader = new CpuShader(`
/*
#error ignored
*/
#define OLD 1
#include <shared>
#include <shared>
#if defined(OLD)
#error undef did not propagate
#elif ENABLED == 2
void main() { gl_FragColor = vec4(gain(0.25)); }
#else
#error define did not propagate
#endif`, {}, { shared: `
#ifndef SHARED
#define SHARED
#undef OLD
#define BASE 2
#define ENABLED BASE
float gain(float x) { return x * float(ENABLED); }
#endif` });

    expect(shader.run({}, context)!.gl_FragColor).toEqual([.5, .5, .5, .5]);
  });
  it("skips inactive expressions and diagnoses malformed macros and conditionals", () => {
    expect(run(`#if 0
#if unknown()
#elif alsoUnknown()
#endif
#endif
void main() { gl_FragColor = vec4(1.0); }`).gl_FragColor).toEqual([1, 1, 1, 1]);
    expect(() => new CpuShader("#define F(x) x\nvoid main(){ F(1, 2); }")).toThrow("macro arguments");
    expect(() => new CpuShader("#if 1\n#else\n#else\n#endif\nvoid main(){}")).toThrow("conditional");
    expect(() => new CpuShader("#error unavailable\nvoid main(){}")).toThrow("unavailable");
  });
  it("supports nested structs, constructors, copies, arrays and inout members", () => {
    const result = run(`
struct Light { vec3 color; float strength; };
struct Scene { Light lights[2]; };
void brighten(inout Light light) { light.color *= light.strength; }
void main() {
  Scene scene; scene.lights[0] = Light(vec3(0.25), 2.0);
  Scene saved = scene;
  brighten(scene.lights[0]);
  gl_FragColor = vec4(scene.lights[0].color, saved.lights[0].color.x);
}`);

    expect(result.gl_FragColor).toEqual([.5, .5, .5, .25]);
  });
  it("imports matrix arrays and nested matrix uniforms", () => {
    const shader = new CpuShader(`
struct Transform { mat2 basis; };
uniform Transform transforms[2]; uniform mat2 matrices[2];
void main() { gl_FragColor = vec4(transforms[1].basis * vec2(1.0), matrices[0] * vec2(1.0)); }`);

    expect(shader.run({ transforms: [{ basis: [1, 0, 0, 1] }, { basis: [2, 0, 0, 3] }],
      matrices: [[4, 0, 0, 5], [1, 0, 0, 1]] }, context)!.gl_FragColor).toEqual([2, 3, 4, 5]);
  });
  it("supports initialized arrays, inferred lengths and array parameters with copy-in/copy-out", () => {
    expect(run(`
void change(inout float values[3]) { values[1] = 8.0; }
void ignore(float values[3]) { values[0] = 99.0; }
void main() {
  float values[] = float[](1.0, 2.0, 3.0); float saved[3] = values;
  ignore(values); change(values);
  gl_FragColor = vec4(values[0], values[1], saved[1], float(values.length()));
}`).gl_FragColor).toEqual([1, 8, 2, 3]);
    expect(() => run("void main(){float a[2] = float[](1.0);}")).toThrow("length mismatch");
    expect(() => run("void main(){float a[5000];}")).toThrow("array length");
  });
  it("selects overloads by scalar, vector, struct and return types", () => {
    expect(run(`
struct Color { vec3 rgb; };
float f(float x) { return 1.0; }
float f(int x) { return 2.0; }
float f(vec3 x) { return 3.0; }
float f(Color x) { return 4.0; }
vec3 makeColor() { return vec3(1.0); }
void main() { gl_FragColor = vec4(f(1.0), f(1), f(makeColor()), f(Color(vec3(1.0)))); }
`).gl_FragColor).toEqual([1, 2, 3, 4]);
  });
  it("executes do/while, switch fallthrough, break and continue within bounded loops", () => {
    expect(run(`void main(){
  int i=0; float sum=0.0;
  do { i++; switch(i) {
    case 1: sum+=1.0;
    case 2: sum+=2.0; break;
    case 3: continue;
    default: sum+=4.0;
  } sum+=10.0; } while(i<4);
  gl_FragColor=vec4(sum, float(i), 0.0, 1.0);
}`).gl_FragColor).toEqual([39, 4, 0, 1]);
    expect(() => new CpuShader("void main(){do {continue;} while(true);}").run({}, { ...context, budget: 10 })).toThrow("budget exceeded");
  });
  it("reads and writes matrix columns, preserves value semantics and checks bounds", () => {
    const result = run(`uniform mat2 original; void main(){
      mat2 m=mat2(1.0); mat2 saved=m;
      m[0]=vec2(2.0,3.0); m[1][0]=4.0;
      original = (m + mat2(1.0)) / 2.0;
      gl_FragColor=vec4(m*vec2(1.0), saved[0]);
    }`);

    expect(result.gl_FragColor).toEqual([6, 4, 1, 0]);
    expect(result.original).toEqual(matrixValue([1.5, 1.5, 2, 1]));
    expect(() => run("void main(){mat2 m=mat2(1.0); gl_FragColor=vec4(m[2],0.0,1.0);}")).toThrow("matrix index");
  });
  it("implements square matrix functions including pivoting and singular diagnostics", () => {
    expect(run(`void main(){
      mat2 m=mat2(0.0,2.0,1.0,3.0);
      mat2 identity=m*inverse(m);
      gl_FragColor=vec4(identity[0], determinant(m), transpose(m)[0][1]);
    }`).gl_FragColor).toEqual([1, 0, -2, 1]);
    expect(run(`void main(){
      mat2 m=matrixCompMult(outerProduct(vec2(2.0,3.0),vec2(4.0,5.0)),mat2(2.0));
      gl_FragColor=vec4(m[0],m[1]);
    }`).gl_FragColor).toEqual([16, 0, 0, 30]);
    expect(() => run("void main(){mat3 m=inverse(mat3(0.0));}")).toThrow("singular");
  });
  it("accepts hexadecimal literals and bitwise assignments", () => {
    expect(run(`void main(){int bits=0xF; bits<<=2; bits|=2; bits&=0x1F; bits^=1; bits>>=1;
      gl_FragColor=vec4(float(bits), float(uint(-1)), float(bool(vec2(0.0,1.0))), 1.0);
    }`).gl_FragColor).toEqual([15, 4294967295, 0, 1]);
  });
  it("evaluates assignment and inout indices once and captures them before a call", () => {
    expect(run(`
void change(inout float value, inout int index) { value=7.0; index=2; }
void main() {
  float a[3]=float[](1.0,2.0,3.0); int i=0;
  a[i++] += 4.0;
  change(a[i++], i);
  gl_FragColor=vec4(a[0],a[1],a[2],float(i));
}`).gl_FragColor).toEqual([5, 7, 3, 2]);
    expect(run(`void main(){mat2 m=mat2(1.0); int i=0; m[i++][1]+=3.0;
      gl_FragColor=vec4(m[0],float(i),1.0);
    }`).gl_FragColor).toEqual([1, 3, 1, 1]);
  });
  it("keeps integer division, unsigned shifts and 32-bit multiplication distinct from float math", () => {
    expect(run(`void main(){uint x=0xffffffffu; x*=0xffffffffu;
      gl_FragColor=vec4(float(7/2), 7.0/2.0, float(0xffffffffu>>31), float(x));
    }`).gl_FragColor).toEqual([3, 3.5, 1, 1]);
    expect(run(`void main(){uint x=0xffffffffu; x++;
      gl_FragColor=vec4(float(~0u),float(-1u),float(x),float(-7/2));
    }`).gl_FragColor).toEqual([4294967295, 4294967295, 0, -3]);
  });
});
