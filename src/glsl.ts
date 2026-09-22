import type { TextureSampling } from "./texture.js";

/** A bounded GLSL interpreter. No eval/Function, DOM access or executable payloads cross the worker boundary. */
export type Value = number | boolean | string | null | Value[] | {
  [key: string]: Value;
};

type Expr = {
  op: string;
  args: Expr[];
  name?: string;
  value?: Value;
  type?: string;
};

type Statement = {
  kind: string;
  expressions?: Expr[];
  body?: Statement[];
  alternate?: Statement;
  name?: string;
  type?: string;
  qualifier?: string;
};

interface FunctionDefinition {
  returnType: string;
  parameters: {
    name: string;
    type: string;
    qualifier: string;
    size?: Expr;
  }[];
  body: Statement;
}

export class ShaderError extends Error {
  constructor(message: string) {
    super(`CPU GLSL: ${message}`);
    this.name = "ShaderError";
  }
}

const qualifiers = new Set([
  "const",
  "uniform",
  "varying",
  "attribute",
  "in",
  "out",
  "inout",
  "highp",
  "mediump",
  "lowp",
  "flat",
  "smooth",
  "centroid",
]);

const types = new Set([
  "void",
  "float",
  "int",
  "uint",
  "bool",
  "vec2",
  "vec3",
  "vec4",
  "ivec2",
  "ivec3",
  "ivec4",
  "uvec2",
  "uvec3",
  "uvec4",
  "bvec2",
  "bvec3",
  "bvec4",
  "mat2",
  "mat3",
  "mat4",
  "sampler2D",
]);

const precedence: Record<string, number> = {
  "=": 1,
  "+=": 1,
  "-=": 1,
  "*=": 1,
  "/=": 1,
  "%=": 1,
  "&=": 1,
  "|=": 1,
  "^=": 1,
  "<<=": 1,
  ">>=": 1,
  "?": 2,
  "||": 3,
  "^^": 4,
  "&&": 5,
  "|": 6,
  "^": 7,
  "&": 8,
  "==": 9,
  "!=": 9,
  "<": 10,
  ">": 10,
  "<=": 10,
  ">=": 10,
  "<<": 11,
  ">>": 11,
  "+": 12,
  "-": 12,
  "*": 13,
  "/": 13,
  "%": 13,
};

const literal = (value: Value): Expr => ({
  op: "literal",
  value,
  args: [],
});

function tokenize(source: string): string[] {
  const clean = source.replace(/\/\*[\s\S]*?\*\//g, " ").replace(/\/\/[^\n]*/g, " ");
  const tokens: string[] = [];
  const pattern = /\s+|0[xX][\da-fA-F]+[uU]?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?|[A-Za-z_]\w*|<<=|>>=|\+\+|--|[+\-*/%=!<>&|^]=|&&|\|\||\^\^|<<|>>|[{}()[\].,;?:+\-*/%!=<>~&|^]/y;
  let index = 0;

  while (index < clean.length) {
    pattern.lastIndex = index;
    const match = pattern.exec(clean);
    if (!match)
      throw new ShaderError(`unexpected token near '${clean.slice(index, index + 32)}'`);
    index = pattern.lastIndex;
    if (match[0].trim())
      tokens.push(match[0]);
  }

  return tokens;
}

class Parser {
  index = 0;
  functions = new Map<string, FunctionDefinition[]>();
  globals: Statement[] = [];
  varyings: string[] = [];
  outputs: string[] = [];
  structs = new Map<string, Statement[]>();

  isType(name: string) {
    return types.has(name) || this.structs.has(name);
  }

  constructor(readonly tokens: string[]) { }

  peek(offset = 0) {
    return this.tokens[this.index + offset] ?? "";
  }

  take() {
    if (!this.peek())
      throw new ShaderError("unexpected end of source");

    return this.tokens[this.index++];
  }

  match(token: string) {
    if (this.peek() !== token)
      return false;
    this.index++;

    return true;
  }

  expect(token: string) {
    if (!this.match(token))
      throw new ShaderError(`expected '${token}', got '${this.peek()}'`);
  }

  identifier() {
    const name = this.take();
    if (!/^[A-Za-z_]\w*$/.test(name))
      throw new ShaderError(`expected identifier, got '${name}'`);

    return name;
  }

  declarationHeader() {
    let qualifier = "";

    while (qualifiers.has(this.peek())) {
      const q = this.take();
      if (["uniform", "varying", "attribute", "in", "out", "inout"].includes(q))
        qualifier = q;
    }

    const type = this.take();
    if (!this.isType(type))
      throw new ShaderError(`unsupported type '${type}'`);

    return {
      type,
      qualifier,
    };
  }

  parse() {
    while (this.peek()) {
      if (this.match("struct")) {
        const name = this.identifier();
        if (this.isType(name)) throw new ShaderError(`duplicate type '${name}'`);
        this.expect("{");
        const fields: Statement[] = [];

        while (this.peek() && this.peek() !== "}") {
          const { type, qualifier } = this.declarationHeader();
          fields.push(...this.declarations(type, qualifier, this.identifier()).body!);
        }

        this.expect("}");
        this.structs.set(name, fields);
        if (!this.match(";"))
          this.globals.push(...this.declarations(name, "", this.identifier()).body!);
        continue;
      }

      if (this.match("precision")) {
        this.take();
        this.take();
        this.expect(";");
        continue;
      }

      const { type, qualifier } = this.declarationHeader();
      const name = this.identifier();

      if (this.match("(")) {
        const parameters: FunctionDefinition["parameters"] = [];
        if (this.peek() === "void" && this.peek(1) === ")")
          this.take();
        else if (this.peek() !== ")")
          do {
            const h = this.declarationHeader();
            const n = this.identifier();
            let size: Expr | undefined;

            if (this.match("[")) {
              size = this.peek() === "]" ? literal(0) : this.expression();
              this.expect("]");
            }

            parameters.push({
              ...h,
              name: n,
              size,
            });
          } while (this.match(","));
        this.expect(")");
        if (this.match(";"))
          continue;
        const body = this.statement();
        const overloads = this.functions.get(name) ?? [];
        overloads.push({
          returnType: type,
          parameters,
          body,
        });
        this.functions.set(name, overloads);
      } else {
        const declarations = this.declarations(type, qualifier, name);
        this.globals.push(...declarations.body!);
        if (qualifier === "varying" || qualifier === "out" || qualifier === "in")
          this.varyings.push(...declarations.body!.map(s => s.name!));
        if (qualifier === "out")
          this.outputs.push(...declarations.body!.map(s => s.name!));
      }
    }

    if (!this.functions.has("main"))
      throw new ShaderError("missing main()");

    return this;
  }

  declarations(type: string, qualifier: string, firstName: string): Statement {
    const body: Statement[] = [];
    let name = firstName;

    do {
      let size: Expr | undefined;

      if (this.match("[")) {
        size = this.peek() === "]" ? literal(0) : this.expression();
        this.expect("]");
      }

      const init = this.match("=") ? this.expression(2) : {

        op: "default",
        name: type,
        args: [],

      };

      body.push({

        kind: "declare",
        name,
        type,
        qualifier,
        expressions: [init, ...(size ? [size] : [])],

      });
      if (!this.match(","))
        break;
      name = this.identifier();
    } while (this.peek());

    this.expect(";");

    return {
      kind: "sequence",
      body,
    };
  }

  statement(): Statement {
    if (this.match("{")) {
      const body: Statement[] = [];
      while (this.peek() && this.peek() !== "}")
        body.push(this.statement());
      this.expect("}");

      return {
        kind: "block",
        body,
      };
    }

    if (this.match(";"))
      return {
        kind: "sequence",
        body: [],
      };

    if (this.match("if")) {
      this.expect("(");
      const condition = this.expression();
      this.expect(")");
      const body = [this.statement()];
      const alternate = this.match("else") ? this.statement() : undefined;

      return {

        kind: "if",
        expressions: [condition],
        body,
        alternate,

      };
    }

    if (this.match("for")) {
      this.expect("(");
      const init = this.statement();
      const test = this.peek() === ";" ? literal(true) : this.expression();
      this.expect(";");
      const increment = this.peek() === ")" ? literal(0) : this.expression();
      this.expect(")");

      return {

        kind: "for",
        body: [init, this.statement()],
        expressions: [test, increment],

      };
    }

    if (this.match("while")) {
      this.expect("(");
      const test = this.expression();
      this.expect(")");

      return {

        kind: "while",
        expressions: [test],
        body: [this.statement()],

      };
    }

    if (this.match("do")) {
      const body = [this.statement()];
      this.expect("while");
      this.expect("(");
      const test = this.expression();
      this.expect(")");
      this.expect(";");

      return { kind: "do", expressions: [test], body };
    }

    if (this.match("switch")) {
      this.expect("(");
      const selector = this.expression();
      this.expect(")");
      this.expect("{");
      const body: Statement[] = [];
      let branch: Statement | undefined;
      let hasDefault = false;

      while (this.peek() && this.peek() !== "}") {
        if (this.match("case")) {
          branch = { kind: "case", expressions: [this.expression()], body: [] };
          this.expect(":");
          body.push(branch);
        } else if (this.match("default")) {
          if (hasDefault) throw new ShaderError("duplicate switch default");
          hasDefault = true;
          branch = { kind: "case", body: [] };
          this.expect(":");
          body.push(branch);
        } else {
          if (!branch) throw new ShaderError("expected switch label");
          branch.body!.push(this.statement());
        }
      }

      this.expect("}");

      return { kind: "switch", expressions: [selector], body };
    }

    if (["return", "discard", "break", "continue"].includes(this.peek())) {
      const kind = this.take();
      const expressions = kind === "return" && this.peek() !== ";" ? [this.expression()] : [];
      this.expect(";");

      return {
        kind,
        expressions,
      };
    }

    if (this.isType(this.peek()) || qualifiers.has(this.peek())) {
      const { type, qualifier } = this.declarationHeader();

      return this.declarations(type, qualifier, this.identifier());
    }

    const expression = this.expression();
    this.expect(";");

    return {
      kind: "expression",
      expressions: [expression],
    };
  }

  expression(min = 1): Expr {
    let left: Expr;
    const token = this.take();
    if (["-", "+", "!", "~", "++", "--"].includes(token))
      left = {
        op: `pre${token}`,
        args: [this.expression(14)],
      };
    else if (token === "(") {
      left = this.expression();
      this.expect(")");
    } else if (/^(?:\d|\.\d)/.test(token))
      left = { ...literal(Number(token.replace(/^0[xX]/.test(token) ? /[uU]$/ : /[fFuU]$/, ""))),
        type: /[uU]$/.test(token) ? "uint" : !/^0[xX]/.test(token) && /[.eEfF]/.test(token) ? "float" : "int" };
    else if (token === "true" || token === "false")
      left = literal(token === "true");
    else if (/^[A-Za-z_]\w*$/.test(token))
      left = {

        op: "name",
        name: token,
        args: [],

      };
    else
      throw new ShaderError(`unexpected expression '${token}'`);

    while (true) {
      if (this.match(".")) {
        left = {

          op: "member",
          name: this.identifier(),
          args: [left],

        };
        continue;
      }

      if (this.match("[")) {
        const index = this.peek() === "]" ? literal(0) : this.expression();
        this.expect("]");
        left = {
          op: "index",
          args: [left, index],
        };
        continue;
      }

      if (this.match("(")) {
        const args: Expr[] = [];
        if (this.peek() !== ")")
          do {
            args.push(this.expression(2));
          } while (this.match(","));
        this.expect(")");

        if (left.op === "member" && left.name === "length" && args.length === 0) {
          left = { op: "arrayLength", args: left.args };
          continue;
        }

        if (left.op === "index" && left.args[0].op === "name" && this.isType(left.args[0].name!)) {
          left = { op: "arrayConstructor", name: left.args[0].name, args: [left.args[1], ...args] };
          continue;
        }

        if (left.op !== "name")
          throw new ShaderError("indirect calls are unsupported");
        left = {

          op: "call",
          name: left.name,
          args,

        };
        continue;
      }

      if (this.peek() === "++" || this.peek() === "--") {
        left = {
          op: `post${this.take()}`,
          args: [left],
        };
        continue;
      }

      const op = this.peek(), level = precedence[op];
      if (!level || level < min)
        break;
      this.take();
      if (op === "?") {
        const yes = this.expression();
        this.expect(":");
        left = {
          op,
          args: [left, yes, this.expression(2)],
        };
      } else
        left = {
          op,
          args: [left, this.expression(level === 1 ? 1 : level + 1)],
        };
    }

    return left;
  }
}

const copy = (v: Value): Value =>
  Array.isArray(v) ? v.map(copy) : v && typeof v === "object" ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, copy(x)])) : v;

const array = (v: Value): Value[] => Array.isArray(v) ? v : [v];
const num = (v: Value): number => Number(v);

// The loops below are the shader's hottest path: no closures or intermediate per-component arrays.
function map(v: Value, fn: (x: number) => number | boolean): Value {
  if (typeof v === "number")
    return fn(v);
  if (isMatrix(v))
    return { matrix: map(v.matrix, fn), size: v.size };

  if (Array.isArray(v)) {
    const out = new Array<Value>(v.length);

    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      out[i] = typeof x === "number" ? fn(x) : map(x, fn);
    }

    return out;
  }

  return fn(num(v));
}

function zip(a: Value, b: Value, fn: (x: number, y: number) => number | boolean): Value {
  if (typeof a === "number" && typeof b === "number")
    return fn(a, b);

  if (isMatrix(a) || isMatrix(b)) {
    if (isMatrix(a) && isMatrix(b) && a.size !== b.size)
      throw new ShaderError("matrix dimensions do not match");

    return { matrix: zip(isMatrix(a) ? a.matrix : a, isMatrix(b) ? b.matrix : b, fn),
      size: isMatrix(a) ? a.size : (b as { size: number }).size };
  }

  if (Array.isArray(a)) {
    const n = a.length;

    if (Array.isArray(b)) {
      if (n !== 1 && b.length !== 1 && n !== b.length)
        throw new ShaderError("vector dimensions do not match");
      const m = Math.max(n, b.length), out = new Array<Value>(m);

      for (let i = 0; i < m; i++) {
        const x = a[n === 1 ? 0 : i], y = b[b.length === 1 ? 0 : i];
        out[i] = typeof x === "number" && typeof y === "number" ? fn(x, y) : zip(x, y, fn);
      }

      return out;
    }

    const y = num(b), out = new Array<Value>(n);

    for (let i = 0; i < n; i++) {
      const x = a[i];
      out[i] = typeof x === "number" ? fn(x, y) : zip(x, b, fn);
    }

    return out;
  }

  if (Array.isArray(b)) {
    const x = num(a), out = new Array<Value>(b.length);

    for (let i = 0; i < b.length; i++) {
      const y = b[i];
      out[i] = typeof y === "number" ? fn(x, y) : zip(a, y, fn);
    }

    return out;
  }

  return fn(num(a), num(b));
}

export const matrixValue = (data: number[]): Value => ({
  matrix: [...data],
  size: Math.sqrt(data.length),
});

const isMatrix = (v: Value): v is {
  matrix: Value[];
  size: number;
} => !!v && typeof v === "object" && !Array.isArray(v) && "matrix" in v;

const times = (x: number, y: number) => x * y;

function multiply(a: Value, b: Value): Value {
  if (typeof a === "number" && typeof b === "number")
    return a * b;
  const am = isMatrix(a), bm = isMatrix(b);
  if (!am && !bm)
    return zip(a, b, times);

  if (am && bm) {
    const n = a.size;
    if (n !== b.size)
      throw new ShaderError("matrix dimensions do not match");
    const out = Array<number>(n * n).fill(0);
    for (let col = 0; col < n; col++)
      for (let row = 0; row < n; row++)
        for (let k = 0; k < n; k++)
          out[col * n + row] += num(a.matrix[k * n + row]) * num(b.matrix[col * n + k]);

    return matrixValue(out);
  }

  if (am && Array.isArray(b)) {
    if (a.size !== b.length) throw new ShaderError("matrix dimensions do not match");
    const n = a.size, m = a.matrix, out = new Array<number>(n);

    for (let row = 0; row < n; row++) {
      let sum = 0;
      for (let c = 0; c < b.length; c++)
        sum += num(m[c * n + row]) * num(b[c]);
      out[row] = sum;
    }

    return out;
  }

  if (Array.isArray(a) && bm) {
    if (a.length !== b.size) throw new ShaderError("matrix dimensions do not match");
    const n = b.size, m = b.matrix, out = new Array<number>(n);

    for (let col = 0; col < n; col++) {
      let sum = 0;
      for (let r = 0; r < a.length; r++)
        sum += num(m[col * n + r]) * num(a[r]);
      out[col] = sum;
    }

    return out;
  }

  if (am)
    return { matrix: map(a.matrix, v => v * num(b)), size: a.size };

  return multiply(b, a);
}

function dot(a: Value, b: Value): number {
  if (!Array.isArray(a))
    return num(a) * num(array(b)[0]);
  const bb = array(b);
  let sum = 0;
  for (let i = 0; i < a.length; i++)
    sum += num(a[i]) * num(bb[i]);

  return sum;
}

const normalize = (v: Value) => map(v, x => x / (Math.sqrt(dot(v, v)) || 1));

/** Size and family (float, int, uint, bool) of each vector type. */
const vecInfo: Record<string, { n: number; kind: string }> = Object.fromEntries(
  ["vec", "ivec", "uvec", "bvec"].flatMap(prefix => [2, 3, 4].map(n => [`${prefix}${n}`, { n, kind: prefix[0] }])));

function construct(type: string, args: Value[]): Value {
  if (type === "float") {
    const v = args[0];

    return typeof v === "number" ? v : num(array(v ?? 0)[0]);
  }

  if (type === "int" || type === "uint")
    return type === "uint" ? num(array(args[0] ?? 0)[0]) >>> 0 : num(array(args[0] ?? 0)[0]) | 0;
  if (type === "bool")
    return !!array(args[0] ?? 0)[0];
  if (type === "sampler2D")
    return args[0] ?? null;

  if (type.startsWith("mat")) {
    const n = Number(type.slice(-1)), out = Array<number>(n * n).fill(0);

    if (isMatrix(args[0])) {
      const m = args[0];
      for (let c = 0; c < n; c++)
        for (let r = 0; r < n; r++)
          out[c * n + r] = c < m.size && r < m.size ? num(m.matrix[c * m.size + r]) : c === r ? 1 : 0;
    } else if (args.length <= 1 && !Array.isArray(args[0])) {
      for (let i = 0; i < n; i++)
        out[i * n + i] = num(args[0] ?? 0);
    } else {
      const flat = args.flatMap(array);
      if (flat.length !== n * n)
        throw new ShaderError(`invalid ${type} constructor`);

      return matrixValue(flat.map(num));
    }

    return matrixValue(out);
  }

  const vec = vecInfo[type];

  if (vec) {
    const { n, kind } = vec;
    // Flattens the arguments in a loop: vec4(vec3, float) and vec3(float) are the usual cases.
    let flat: Value[];

    if (args.length === 1 && Array.isArray(args[0]))
      flat = args[0];
    else {
      flat = [];

      for (const arg of args) {
        if (Array.isArray(arg))
          for (const x of arg) flat.push(x);
        else
          flat.push(arg);
      }
    }

    if (flat.length > 1 && flat.length < n)
      throw new ShaderError(`invalid ${type} constructor`);
    const out = new Array<Value>(n);

    for (let i = 0; i < n; i++) {
      const v = flat.length ? flat[flat.length === 1 ? 0 : i] : 0;
      out[i] = kind === "v" ? (typeof v === "number" ? v : num(v)) : kind === "b" ? !!v : kind === "u" ? num(v) >>> 0 : num(v) | 0;
    }

    return out;
  }

  return 0;
}

export interface ShaderContext {
  texture: (sampler: Value, uv: number[], sampling?: TextureSampling) => number[];
  /** Avoid quad gradients for samplers that only need their magnification filter. */
  textureNeedsFootprint?: (sampler: Value) => boolean;
  budget?: number;
  /** Supplied by fragment quad execution; absent in vertex and scalar evaluations. */
  derivatives?: DerivativeContext;
}

export interface DerivativeContext {
  /** `implicit` marks the derivatives a texture takes for its mip level, which the shader never wrote. */
  evaluate(site: symbol, operation: string, value: Value, implicit?: boolean): Value;
  /** Shared instruction allowance across all helper invocations and replays. */
  budget: { remaining: number };
}

/**
 * Quad lanes are ordered in GL window coordinates: bottom-left, bottom-right, top-left, top-right.
 * Rendezvous replay preserves ordinary scalar interpreter semantics (locals, calls, inout and loops).
 * Only derivative shaders enter this path; each replay starts with isolated inputs.
 */
export function evaluateDerivativeQuad<T>(evaluate: (lane: number, context: DerivativeContext) => T,
  selected?: number, budget = 200000): T[] {
  const resolved: { site: symbol; operation: string; values: Value[] }[] = [];
  const allowance = { remaining: budget };
  const pause = Symbol("derivative rendezvous");
  const divergent = "derivative in non-uniform control flow or after divergent discard";
  const order = selected === undefined ? [0, 1, 2, 3] : [selected, ...[0, 1, 2, 3].filter(i => i !== selected)];
  const results: T[] = new Array(4);

  const difference = (a: Value, b: Value): Value => {
    if (typeof a === "number" && typeof b === "number") return b - a;
    if (Array.isArray(a) && Array.isArray(b) && a.length === b.length)
      return a.map((v, i) => difference(v, b[i]));
    throw new ShaderError("derivatives require matching float or vector operands");
  };

  const width = (a: Value, b: Value): Value => Array.isArray(a)
    ? a.map((v, i) => width(v, (b as Value[])[i])) : Math.abs(a as number) + Math.abs(b as number);

  /**
   * Lanes diverged where only implicit texture derivatives were pending: like a GPU, the quad keeps going.
   * Each lane replays the derivatives resolved so far and gets zero for the rest, so textures sample the base
   * level. A derivative the shader wrote still requires uniform control flow.
   */
  const withoutDerivatives = (): T[] => {
    for (const lane of order) {
      let cursor = 0;

      results[lane] = evaluate(lane, {
        budget: allowance,
        evaluate(site, operation, value, implicit) {
          const known = resolved[cursor++];

          if (known) {
            if (known.site !== site || known.operation !== operation)
              throw new ShaderError("derivative in non-uniform control flow");

            return copy(known.values[lane]);
          }

          if (!implicit)
            throw new ShaderError(divergent);

          return difference(value, value);
        },
      });
      if (lane === selected) return results;
    }

    return results;
  };

  for (let round = 0; round < 1024; round++) {
    const pending: { site: symbol; operation: string; value: Value; implicit: boolean }[] = [];
    let completed = 0;

    for (const lane of order) {
      let cursor = 0;

      try {
        results[lane] = evaluate(lane, {
          budget: allowance,
          evaluate(site, operation, value, implicit) {
            const known = resolved[cursor++];

            if (known) {
              if (known.site !== site || known.operation !== operation)
                throw new ShaderError("derivative in non-uniform control flow");

              return copy(known.values[lane]);
            }

            pending[lane] = { site, operation, value: copy(value), implicit: implicit === true };
            throw pause;
          },
        });
        completed++;
        if (lane === selected) return results;
      } catch(error) {
        if (error !== pause) throw error;
      }
    }

    if (completed === 4) return results;
    const first = pending[0];

    if (completed || !first || pending.some(p => p.site !== first.site || p.operation !== first.operation)) {
      if (pending.some(Boolean) && pending.every(p => !p || p.implicit))
        return withoutDerivatives();
      throw new ShaderError(divergent);
    }

    const values = pending.map(p => p.value);
    const dx = [difference(values[0], values[1]), difference(values[2], values[3])];
    const dy = [difference(values[0], values[2]), difference(values[1], values[3])];
    resolved.push({ site: first.site, operation: first.operation, values: values.map((_, lane) =>
      first.operation === "dFdx" ? dx[lane >> 1] : first.operation === "dFdy" ? dy[lane & 1]
        : width(dx[lane >> 1], dy[lane & 1])) });
  }

  throw new ShaderError("derivative rendezvous budget exceeded");
}

const swizzle = (name: string) =>
  [...name].map(c =>
    "xyzw".includes(c) ? "xyzw".indexOf(c) : "rgba".includes(c) ? "rgba".indexOf(c) : "stpq".indexOf(c));

const DISCARD = Symbol("discard");
// Control flow returned by each compiled statement.
const FLOW_NONE = 0, FLOW_BREAK = 1, FLOW_CONTINUE = 2, FLOW_RETURN = 3, FLOW_DISCARD = 4;

interface Runtime {
  remaining: number;
  depth: number;
  returned: Value;
  context: ShaderContext;
}

interface Frame {
  locals: Value[];
  globals: Value[];
  rt: Runtime;
}

type Thunk = (f: Frame) => Value;

interface Reference { read: () => Value; write: (value: Value) => void }

type Accessor = (f: Frame) => Reference;

type Exec = (f: Frame) => number;

interface CompiledFunction {
  returnType: string;
  parameters: FunctionDefinition["parameters"];
  slots: number;
  body: Exec;
}

interface FunctionState {
  slots: number;
}

function tick(rt: Runtime) {
  if ((rt.context.derivatives && --rt.context.derivatives.budget.remaining <= 0) || --rt.remaining <= 0)
    throw new ShaderError("instruction budget exceeded");
}

/** Lexical compile-time scope: local names become fixed indices in the function's frame. */
class CompileScope {
  private names = new Map<string, number>();
  private types = new Map<string, string>();

  constructor(private parent: CompileScope | null, private fn: FunctionState) { }

  child() {
    return new CompileScope(this, this.fn);
  }

  declare(name: string, type = "") {
    const slot = this.fn.slots++;
    this.names.set(name, slot);
    this.types.set(name, type);

    return slot;
  }

  resolve(name: string): number | undefined {
    return this.names.get(name) ?? this.parent?.resolve(name);
  }

  type(name: string): string | undefined {
    return this.types.get(name) ?? this.parent?.type(name);
  }
}

const binaryOps: Record<string, (a: Value, b: Value) => Value> = {
  "*": multiply,
  "+": (a, b) => zip(a, b, (x, y) => x + y),
  "-": (a, b) => zip(a, b, (x, y) => x - y),
  "/": (a, b) => zip(a, b, (x, y) => x / y),
  "%": (a, b) => zip(a, b, (x, y) => x % y),
  "<": (a, b) => zip(a, b, (x, y) => x < y),
  ">": (a, b) => zip(a, b, (x, y) => x > y),
  "<=": (a, b) => zip(a, b, (x, y) => x <= y),
  ">=": (a, b) => zip(a, b, (x, y) => x >= y),
  "&": (a, b) => zip(a, b, (x, y) => x & y),
  "|": (a, b) => zip(a, b, (x, y) => x | y),
  "^": (a, b) => zip(a, b, (x, y) => x ^ y),
  "<<": (a, b) => zip(a, b, (x, y) => x << y),
  ">>": (a, b) => zip(a, b, (x, y) => x >> y),
  "^^": (a, b) => zip(a, b, (x, y) => !!x !== !!y),
  "==": (a, b) => JSON.stringify(a) === JSON.stringify(b),
  "!=": (a, b) => JSON.stringify(a) !== JSON.stringify(b),
};

const unary: Record<string, (x: number) => number> = {
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  sinh: Math.sinh,
  cosh: Math.cosh,
  tanh: Math.tanh,
  asin: Math.asin,
  acos: Math.acos,
  abs: Math.abs,
  sign: Math.sign,
  floor: Math.floor,
  ceil: Math.ceil,
  round: Math.round,
  trunc: Math.trunc,
  sqrt: Math.sqrt,
  inversesqrt: x => 1 / Math.sqrt(x),
  exp: Math.exp,
  exp2: x => 2 ** x,
  log: Math.log,
  log2: Math.log2,
  fract: x => x - Math.floor(x),
  radians: x => x * Math.PI / 180,
  degrees: x => x * 180 / Math.PI,
};

const zipped: Record<string, (x: number, y: number) => number> = {
  pow: Math.pow,
  min: Math.min,
  max: Math.max,
  mod: (x, y) => x - y * Math.floor(x / y),
  step: (edge, x) => x < edge ? 0 : 1,
};

const compared: Record<string, (x: number, y: number) => boolean> = {
  lessThan: (x, y) => x < y,
  greaterThan: (x, y) => x > y,
  lessThanEqual: (x, y) => x <= y,
  greaterThanEqual: (x, y) => x >= y,
  equal: (x, y) => x === y,
  notEqual: (x, y) => x !== y,
};

const minus = (x: number, y: number) => x - y;
const saturateRatio = (x: number, y: number) => Math.max(0, Math.min(1, x / y));
const smooth = (x: number) => x * x * (3 - 2 * x);
const oneMinus = (x: number) => 1 - x;

/** Built-in functions resolved at compile time; null when the name is not a built-in. */
function compileBuiltin(name: string, args: Thunk[]): Thunk | null {
  const [a, b, c] = args;
  if (unary[name])
    return f => map(a(f), unary[name]);
  if (zipped[name])
    return f => zip(a(f), b(f), zipped[name]);
  if (compared[name])
    return f => zip(a(f), b(f), compared[name]);

  switch (name) {
    case "transpose": return f => {
      const m = a(f);
      if (!isMatrix(m)) throw new ShaderError("transpose requires a matrix");
      const out = new Array<number>(m.size * m.size);
      for (let col = 0; col < m.size; col++)
        for (let row = 0; row < m.size; row++) out[col * m.size + row] = num(m.matrix[row * m.size + col]);

      return matrixValue(out);
    };

    case "matrixCompMult": return f => {
      const x = a(f), y = b(f);
      if (!isMatrix(x) || !isMatrix(y)) throw new ShaderError("matrixCompMult requires matrices");

      return zip(x, y, times);
    };

    case "outerProduct": return f => {
      const x = array(a(f)), y = array(b(f));
      if (x.length !== y.length) throw new ShaderError("non-square matrices are unsupported");

      return matrixValue(y.flatMap(col => x.map(row => num(row) * num(col))));
    };

    case "determinant":
    case "inverse": return f => {
      const matrix = a(f);
      if (!isMatrix(matrix)) throw new ShaderError(`${name} requires a matrix`);
      const n = matrix.size, m = matrix.matrix.map(num), inverse = construct(`mat${n}`, [1]) as { matrix: number[]; size: number };
      let det = 1;

      for (let k = 0; k < n; k++) {
        let pivot = k;
        for (let row = k + 1; row < n; row++)
          if (Math.abs(m[k * n + row]) > Math.abs(m[k * n + pivot])) pivot = row;

        if (m[k * n + pivot] === 0) {
          if (name === "determinant") return 0;
          throw new ShaderError("cannot invert singular matrix");
        }

        if (pivot !== k) {
          det = -det;

          for (let col = 0; col < n; col++) {
            [m[col * n + k], m[col * n + pivot]] = [m[col * n + pivot], m[col * n + k]];
            [inverse.matrix[col * n + k], inverse.matrix[col * n + pivot]] =
              [inverse.matrix[col * n + pivot], inverse.matrix[col * n + k]];
          }
        }

        const diagonal = m[k * n + k];
        det *= diagonal;

        for (let col = 0; col < n; col++) {
          m[col * n + k] /= diagonal;
          inverse.matrix[col * n + k] /= diagonal;
        }

        for (let row = 0; row < n; row++) if (row !== k) {
          const factor = m[k * n + row];

          for (let col = 0; col < n; col++) {
            m[col * n + row] -= factor * m[col * n + k];
            inverse.matrix[col * n + row] -= factor * inverse.matrix[col * n + k];
          }
        }
      }

      return name === "determinant" ? det : inverse;
    };

    case "atan": return args.length === 2 ? f => zip(a(f), b(f), Math.atan2) : f => map(a(f), Math.atan);
    case "clamp": return f => zip(zip(a(f), b(f), Math.max), c(f), Math.min);

    case "mix": return f => {
      const t = c(f);

      return binaryOps["+"](multiply(a(f), map(t, oneMinus)), multiply(b(f), t));
    };

    case "smoothstep": return f => {
      const lo = a(f), hi = b(f);
      const t = zip(zip(c(f), lo, minus), zip(hi, lo, minus), saturateRatio);

      return map(t, smooth);
    };

    case "dot": return f => dot(a(f), b(f));

    case "length": return f => {
      const v = a(f);

      return Math.sqrt(dot(v, v));
    };

    case "distance": return f => {
      const v = binaryOps["-"](a(f), b(f));

      return Math.sqrt(dot(v, v));
    };

    case "normalize": return f => normalize(a(f));

    case "cross": return f => {
      const u = array(a(f)).map(num), v = array(b(f)).map(num);

      return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
    };

    case "reflect": return f => {
      const i = a(f), n = b(f);

      return binaryOps["-"](i, multiply(n, 2 * dot(n, i)));
    };

    case "refract": return f => {
      const i = a(f), n = b(f), eta = num(c(f)), d = dot(n, i), k = 1 - eta * eta * (1 - d * d);

      return k < 0 ? map(i, () => 0) : binaryOps["-"](multiply(i, eta), multiply(n, eta * d + Math.sqrt(k)));
    };

    case "faceforward": return f => {
      const n = a(f);

      return dot(c(f), b(f)) < 0 ? n : map(n, x => -x);
    };

    case "any": return f => array(a(f)).some(Boolean);
    case "all": return f => array(a(f)).every(Boolean);
    case "not": return f => map(a(f), x => !x);
    default: return null;
  }
}

/** Compiles the tree into closures once per shader; each fragment only runs the closures. */
class CompiledProgram {
  usesDerivatives = false;
  usesImplicitTextureSampling = false;
  private structs = new Map<string, Statement[]>();
  private globalTypes = new Map<string, string>();
  private matrixInputTypes = new Map<string, string>();
  private globalSlots = new Map<string, number>();
  private declarations: { slot: number; exec: Exec }[] = [];
  private functions = new Map<string, CompiledFunction[]>();
  private main: CompiledFunction | null = null;

  constructor(parsed: Parser | null) {
    if (!parsed)
      return;
    this.structs = parsed.structs;
    // Functions enter the map before their bodies: a call may refer to a function defined later.
    for (const [name, overloads] of parsed.functions)
      this.functions.set(name, overloads.map(definition =>
        ({ returnType: definition.returnType, parameters: definition.parameters, slots: 0, body: () => 0 })));
    for (const declaration of parsed.globals)
      this.globalTypes.set(declaration.name!, declaration.type! + (declaration.expressions?.[1] ? "[]" : ""));
    for (const [name, type] of this.globalTypes)
      if (this.needsMatrixImport(type)) this.matrixInputTypes.set(name, type);
    const globalScope = new CompileScope(null, { slots: 0 });

    for (const declaration of parsed.globals) {
      this.declarations.push({
        slot: this.globalSlot(declaration.name!),
        exec: this.compileDeclare(declaration, globalScope, true),
      });
    }

    for (const [name, overloads] of parsed.functions)
      overloads.forEach((definition, index) => {
        const compiled = this.functions.get(name)![index], state: FunctionState = { slots: 0 };
        const scope = new CompileScope(null, state);
        for (const parameter of definition.parameters)
          scope.declare(parameter.name, parameter.type + (parameter.size ? "[]" : ""));
        compiled.body = this.compileStatement(definition.body, scope);
        compiled.slots = state.slots;
      });
    this.main = this.functions.get("main")?.[0] ?? null;
  }

  private globalSlot(name: string) {
    let slot = this.globalSlots.get(name);

    if (slot === undefined) {
      slot = this.globalSlots.size;
      this.globalSlots.set(name, slot);
    }

    return slot;
  }

  private compileDeclare(statement: Statement, scope: CompileScope, global: boolean): Exec {
    const expressions = statement.expressions ?? [];
    const init = this.compileExpr(expressions[0], scope);
    const size = expressions[1] ? this.compileExpr(expressions[1], scope) : null;
    // The initializer sees the outer name (`float x = x;`), so the slot is only created after it.
    const slot = global ? this.globalSlot(statement.name!) : scope.declare(statement.name!, statement.type! + (size ? "[]" : ""));

    return f => {
      let value = init(f);

      if (size) {
        const count = num(size(f));
        if (!Number.isInteger(count) || count < 0 || count > 4096 || (count === 0 && expressions[0].op === "default"))
          throw new ShaderError("invalid array length");
        if (expressions[0].op === "default")
          value = Array.from({ length: count }, () => copy(value));
        else if (!Array.isArray(value) || (count !== 0 && value.length !== count))
          throw new ShaderError("array initializer length mismatch");
      }

      if (global)
        f.globals[slot] = value;
      else
        f.locals[slot] = value;

      return FLOW_NONE;
    };
  }

  compileStatement(statement: Statement, scope: CompileScope): Exec {
    const { kind, body = [], expressions = [] } = statement;

    switch (kind) {
      case "block":
      case "sequence": {
        const inner = kind === "block" ? scope.child() : scope;
        const steps = body.map(child => this.compileStatement(child, inner));
        if (steps.length === 1)
          return steps[0];

        return f => {
          for (let i = 0; i < steps.length; i++) {
            const flow = steps[i](f);
            if (flow)
              return flow;
          }

          return FLOW_NONE;
        };
      }

      case "declare": return this.compileDeclare(statement, scope, false);

      case "switch": {
        const selector = this.compileExpr(expressions[0], scope), inner = scope.child();

        const branches = body.map(branch => ({
          test: branch.expressions?.[0] ? this.compileExpr(branch.expressions[0], inner) : null,
          exec: this.compileStatement({ kind: "sequence", body: branch.body }, inner),
        }));

        return f => {
          const value = selector(f);
          let start = branches.findIndex(branch => branch.test && branch.test(f) === value);
          if (start < 0) start = branches.findIndex(branch => !branch.test);
          if (start < 0) return FLOW_NONE;

          for (let i = start; i < branches.length; i++) {
            const flow = branches[i].exec(f);
            if (flow === FLOW_BREAK) return FLOW_NONE;
            if (flow) return flow;
          }

          return FLOW_NONE;
        };
      }

      case "expression": {
        const thunk = this.compileExpr(expressions[0], scope);

        return f => {
          thunk(f);

          return FLOW_NONE;
        };
      }

      case "if": {
        const condition = this.compileExpr(expressions[0], scope), then = this.compileStatement(body[0], scope.child());
        const otherwise = statement.alternate ? this.compileStatement(statement.alternate, scope.child()) : null;

        return f => condition(f) ? then(f) : otherwise ? otherwise(f) : FLOW_NONE;
      }

      case "for":
      case "while":
      case "do": {
        const inner = scope.child();
        const init = kind === "for" ? this.compileStatement(body[0], inner) : null;
        const test = this.compileExpr(expressions[0], inner);
        const increment = kind === "for" ? this.compileExpr(expressions[1], inner) : null;
        const loop = this.compileStatement(body[kind === "for" ? 1 : 0], inner);

        return f => {
          init?.(f);
          let first = kind === "do";

          while (first || test(f)) {
            first = false;
            tick(f.rt);
            const flow = loop(f);
            if (flow === FLOW_BREAK)
              break;
            if (flow === FLOW_RETURN || flow === FLOW_DISCARD)
              return flow;
            increment?.(f);
          }

          return FLOW_NONE;
        };
      }

      case "return": {
        const value = expressions[0] ? this.compileExpr(expressions[0], scope) : null;

        return f => {
          f.rt.returned = value ? value(f) : 0;

          return FLOW_RETURN;
        };
      }

      case "discard": return () => FLOW_DISCARD;
      case "break": return () => FLOW_BREAK;
      case "continue": return () => FLOW_CONTINUE;
      default: throw new ShaderError(`unsupported statement ${kind}`);
    }
  }

  private compileReference(expr: Expr, scope: CompileScope): Accessor {
    if (expr.op === "name") {
      const local = scope.resolve(expr.name!);
      const slot = local ?? this.globalSlot(expr.name!);

      return f => {
        const values = local === undefined ? f.globals : f.locals;

        return { read: () => values[slot], write: value => {
          values[slot] = value;
        } };
      };
    }

    if (expr.op !== "member" && expr.op !== "index") throw new ShaderError("invalid assignment target");
    const parent = this.compileReference(expr.args[0], scope);
    const index = expr.op === "index" ? this.compileExpr(expr.args[1], scope) : null;
    const member = expr.name!, indices = expr.op === "member" ? swizzle(member) : [];

    return f => {
      const base = parent(f), i = index ? num(index(f)) : 0;

      const parts = (value: Value): Value[] => {
        const values = isMatrix(value) ? value.matrix : value;
        const length = isMatrix(value) ? value.size : Array.isArray(value) ? value.length : 0;
        if (!Array.isArray(values) || !Number.isInteger(i) || i < 0 || i >= length)
          throw new ShaderError("array or matrix index out of bounds");

        return values;
      };

      return {
        read: () => {
          const value = base.read();

          if (index) {
            const values = parts(value);

            return isMatrix(value) ? values.slice(i * value.size, (i + 1) * value.size) : values[i];
          }

          if (Array.isArray(value)) {
            if (indices.some(i => i < 0 || i >= value.length)) throw new ShaderError("invalid swizzle");

            return indices.length === 1 ? value[indices[0]] : indices.map(i => value[i]);
          }

          if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, member)) return value[member];
          throw new ShaderError("invalid member assignment");
        },
        write: value => {
          const target = copy(base.read());
          if (index) {
            const values = parts(target);
            if (isMatrix(target)) {
              if (!Array.isArray(value) || value.length !== target.size) throw new ShaderError("invalid matrix column assignment");
              values.splice(i * target.size, target.size, ...value);
            } else values[i] = copy(value);
          } else if (Array.isArray(target)) {
            if (new Set(indices).size !== indices.length || indices.some(i => i < 0 || i >= target.length))
              throw new ShaderError("invalid writable swizzle");
            const values = array(value);
            if (values.length !== indices.length) throw new ShaderError("swizzle dimensions do not match");
            indices.forEach((i, k) => {
              target[i] = values[k];
            });
          } else if (target && typeof target === "object" && Object.prototype.hasOwnProperty.call(target, member)) target[member] = value;
          else throw new ShaderError("invalid member assignment");
          base.write(target);
        },
      };
    };
  }

  compileExpr(expr: Expr, scope: CompileScope): Thunk {
    const { op, args, name } = expr;

    if (op === "literal") {
      const value = expr.value!;

      return () => value;
    }

    if (op === "default" && this.structs.has(name!)) {
      const fields = this.structs.get(name!)!;

      const initializers = fields.map(field => {
        const init = this.compileExpr(field.expressions![0], scope);
        const size = field.expressions![1] ? this.compileExpr(field.expressions![1], scope) : null;

        return (f: Frame): Value => {
          const value = init(f);
          if (!size) return value;
          const count = num(size(f));
          if (!Number.isInteger(count) || count < 1 || count > 4096) throw new ShaderError("invalid array length");

          return Array.from({ length: count }, () => copy(value));
        };
      });

      return f => Object.fromEntries(fields.map((field, i) => [field.name!, initializers[i](f)]));
    }

    if (op === "default")
      return () => construct(name!, []);

    if (op === "arrayLength") {
      const base = this.compileExpr(args[0], scope);

      return f => {
        const value = base(f);
        if (isMatrix(value)) return value.size;
        if (!Array.isArray(value)) throw new ShaderError("length() requires an array, vector or matrix");

        return value.length;
      };
    }

    if (op === "arrayConstructor") {
      const size = this.compileExpr(args[0], scope), values = args.slice(1).map(a => this.compileExpr(a, scope));

      return f => {
        const count = num(size(f));
        if ((count !== 0 && count !== values.length) || values.length > 4096)
          throw new ShaderError("array constructor length mismatch");

        return values.map(value => copy(value(f)));
      };
    }

    if (op === "name") {
      const local = scope.resolve(name!);
      if (local !== undefined)
        return f => f.locals[local];
      const slot = this.globalSlot(name!), label = name!;

      return f => {
        const value = f.globals[slot];
        if (value === undefined)
          throw new ShaderError(`unknown variable '${label}'`);

        return value;
      };
    }

    if (op === "member") {
      const base = this.compileExpr(args[0], scope), indices = swizzle(name!), member = name!;
      const single = indices.length === 1 ? indices[0] : -1, valid = indices.every(i => i >= 0);

      return f => {
        const value = base(f);

        if (Array.isArray(value)) {
          if (single >= 0) {
            if (single < value.length)
              return value[single];
          } else if (valid) {
            const out = new Array<Value>(indices.length);

            for (let i = 0; i < indices.length; i++) {
              if (indices[i] >= value.length)
                throw new ShaderError(`invalid swizzle .${member}`);
              out[i] = value[indices[i]];
            }

            return out;
          }

          throw new ShaderError(`invalid swizzle .${member}`);
        }

        if (value && typeof value === "object" && Object.prototype.hasOwnProperty.call(value, member))
          return value[member];
        throw new ShaderError(`unknown member .${member}`);
      };
    }

    if (op === "index") {
      const base = this.compileExpr(args[0], scope), index = this.compileExpr(args[1], scope);

      return f => {
        const value = base(f), i = num(index(f));

        if (isMatrix(value)) {
          if (!Number.isInteger(i) || i < 0 || i >= value.size) throw new ShaderError("matrix index out of bounds");

          return value.matrix.slice(i * value.size, (i + 1) * value.size);
        }

        if (!Array.isArray(value) || !Number.isInteger(i) || i < 0 || i >= value.length)
          throw new ShaderError("array index out of bounds");

        return value[i];
      };
    }

    if (op === "?") {
      const [condition, yes, no] = args.map(a => this.compileExpr(a, scope));

      return f => condition(f) ? yes(f) : no(f);
    }

    if (op === "&&" || op === "||") {
      const [left, right] = args.map(a => this.compileExpr(a, scope));

      return op === "&&" ? f => !!left(f) && !!right(f) : f => !!left(f) || !!right(f);
    }

    if (op === "call")
      return this.compileCall(name!, args, scope);

    if (op === "=" || precedence[op] === 1) {
      const right = this.compileExpr(args[1], scope), access = this.compileReference(args[0], scope);

      if (op === "=")
        return f => {
          const target = access(f), value = right(f);
          target.write(value);

          return value;
        };

      const combine = binaryOps[op.slice(0, -1)];
      const type = this.expressionType(args[0], scope);
      const integer = type && /^(?:u?int|[iu]vec[234])$/.test(type);

      return f => {
        const target = access(f);

        let value = op === ">>=" && type?.startsWith("u")
          ? zip(target.read(), right(f), (x, y) => x >>> y)
          : op === "*=" && integer ? zip(target.read(), right(f), Math.imul) : combine(target.read(), right(f));

        if (integer) value = map(value, x => type!.startsWith("u") ? x >>> 0 : x | 0);
        target.write(value);

        return value;
      };
    }

    if (op === "pre++" || op === "pre--" || op === "post++" || op === "post--") {
      const access = this.compileReference(args[0], scope);
      const delta = op.endsWith("++") ? 1 : -1, post = op.startsWith("post");
      const type = this.expressionType(args[0], scope);

      return f => {
        const target = access(f), before = target.read(), after = map(before, x =>
          type && /^(?:u?int|[iu]vec[234])$/.test(type) ? type.startsWith("u") ? (x + delta) >>> 0 : (x + delta) | 0 : x + delta);

        target.write(after);

        return post ? before : after;
      };
    }

    if (op.startsWith("pre")) {
      const operand = this.compileExpr(args[0], scope);

      const fn = op === "pre-" ? (x: number) => -x : op === "pre!" ? (x: number) => !x
        : op === "pre~" ? (x: number) => ~x : (x: number) => x;

      const type = this.expressionType(expr, scope);
      if (type && /^(?:u?int|[iu]vec[234])$/.test(type))
        return f => map(operand(f), x => type.startsWith("u") ? Number(fn(x)) >>> 0 : Number(fn(x)) | 0);

      return f => map(operand(f), fn);
    }

    const combine = binaryOps[op];
    if (!combine)
      throw new ShaderError(`unsupported operator ${op}`);
    const [left, right] = args.map(a => this.compileExpr(a, scope));
    const type = this.expressionType(expr, scope);
    if (type && /^(?:u?int|[iu]vec[234])$/.test(type))
      return f => map(op === ">>" && type.startsWith("u") ? zip(left(f), right(f), (x, y) => x >>> y)
        : op === "*" ? zip(left(f), right(f), Math.imul) : combine(left(f), right(f)), x => type.startsWith("u") ? x >>> 0 : x | 0);

    return f => combine(left(f), right(f));
  }

  private expressionType(expr: Expr, scope: CompileScope): string | undefined {
    const { op, name, args } = expr;
    if (expr.type) return expr.type;
    if (op === "literal") return typeof expr.value === "boolean" ? "bool" : "float";
    if (op === "name") return scope.type(name!) ?? this.globalTypes.get(name!)
      ?? ({ position: "vec3", normal: "vec3", uv: "vec2", gl_Position: "vec4", gl_FragColor: "vec4",
        gl_FragCoord: "vec4", gl_PointCoord: "vec2", gl_FrontFacing: "bool" } as Record<string, string>)[name!];
    if (op === "default" || op === "arrayConstructor") return name + (op === "arrayConstructor" ? "[]" : "");
    if (op === "arrayLength") return "int";

    if (op === "member") {
      const base = this.expressionType(args[0], scope);
      const field = base && this.structs.get(base)?.find(field => field.name === name);
      if (field) return field.type! + (field.expressions?.[1] ? "[]" : "");

      if (base && /^(?:[iub]?vec)[234]$/.test(base)) {
        const prefix = base.slice(0, -1);

        return name!.length > 1 ? prefix + name!.length : ({ vec: "float", ivec: "int", uvec: "uint", bvec: "bool" })[prefix];
      }

      return undefined;
    }

    if (op === "index") {
      const base = this.expressionType(args[0], scope);
      if (base?.endsWith("[]")) return base.slice(0, -2);
      if (base?.startsWith("mat")) return `vec${base.slice(-1)}`;

      return base && ({ vec: "float", ivec: "int", uvec: "uint", bvec: "bool" })[base.slice(0, -1)];
    }

    if (op === "call") {
      if (types.has(name!) || this.structs.has(name!)) return name;
      if (this.functions.has(name!)) return this.resolveFunction(name!, args, scope)?.returnType;
      if (["dot", "length", "distance", "determinant"].includes(name!)) return "float";
      if (["any", "all"].includes(name!)) return "bool";
      if (["texture", "texture2D", "texture2DProj", "textureLod", "texture2DLod", "texture2DLodEXT",
        "textureGrad", "texture2DGradEXT", "textureProj"].includes(name!)) return "vec4";
      if (name === "outerProduct") return `mat${this.expressionType(args[0], scope)?.slice(-1)}`;
      if (Object.prototype.hasOwnProperty.call(compared, name!))
        return `bvec${this.expressionType(args[0], scope)?.slice(-1)}`;

      return args[0] && this.expressionType(args[0], scope);
    }

    if (["==", "!=", "<", ">", "<=", ">=", "&&", "||", "^^", "pre!"].includes(op)) return "bool";
    if (op === "?") return this.expressionType(args[1], scope);
    const a = args[0] && this.expressionType(args[0], scope), b = args[1] && this.expressionType(args[1], scope);
    if (op === "*" && a?.startsWith("mat") && b?.includes("vec")) return b;
    if (op === "*" && a?.includes("vec") && b?.startsWith("mat")) return a;
    if (b?.includes("vec") || b?.startsWith("mat")) return b;

    return a === "int" && b === "float" ? b : a;
  }

  private resolveFunction(name: string, expressions: Expr[], scope: CompileScope): CompiledFunction | undefined {
    const candidates = this.functions.get(name)?.filter(fn => fn.parameters.length === expressions.length) ?? [];
    if (candidates.length < 2) return candidates[0];
    const argumentTypes = expressions.map(expr => this.expressionType(expr, scope));

    const matches = candidates.filter(fn => fn.parameters.every((p, i) =>
      argumentTypes[i] === p.type + (p.size ? "[]" : "")));

    if (matches.length !== 1) throw new ShaderError(`ambiguous overload '${name}' (${argumentTypes.join(", ")})`);

    return matches[0];
  }

  private compileCall(name: string, expressions: Expr[], scope: CompileScope): Thunk {
    const args = expressions.map(e => this.compileExpr(e, scope));
    const fields = this.structs.get(name);

    if (fields) {
      if (args.length !== fields.length) throw new ShaderError(`invalid ${name} constructor`);

      return f => Object.fromEntries(fields.map((field, i) => [field.name!, copy(args[i](f))]));
    }

    if (types.has(name)) {
      const vec = vecInfo[name];

      // vec3(a, b, c) with scalars already is the vector itself: nothing to flatten or convert.
      if (vec && vec.kind === "v" && args.length === vec.n)
        return f => {
          const values = new Array<Value>(args.length);
          let scalar = true;

          for (let i = 0; i < args.length; i++) {
            const v = args[i](f);
            values[i] = v;
            if (typeof v !== "number")
              scalar = false;
          }

          return scalar ? values : construct(name, values);
        };

      return f => {
        const values = new Array<Value>(args.length);
        for (let i = 0; i < args.length; i++)
          values[i] = args[i](f);

        return construct(name, values);
      };
    }

    const callee = this.resolveFunction(name, expressions, scope);

    if (callee) {
      const setters = callee.parameters.map((p, i) =>
        p.qualifier === "out" || p.qualifier === "inout" ? this.compileReference(expressions[i], scope) : null);

      return f => {
        const rt = f.rt, frame: Frame = { locals: new Array(callee.slots), globals: f.globals, rt };
        const references: (Reference | null)[] = [];

        for (let i = 0; i < args.length; i++) {
          const reference = setters[i]?.(f) ?? null;
          references.push(reference);
          frame.locals[i] = reference ? reference.read() : args[i](f);
        }

        if (++rt.depth > 32)
          throw new ShaderError("call depth exceeded");
        tick(rt);
        const flow = callee.body(frame);
        rt.depth--;
        if (flow === FLOW_DISCARD)
          throw DISCARD;
        const returned = flow === FLOW_RETURN ? rt.returned : 0;
        for (let i = 0; i < references.length; i++)
          references[i]?.write(frame.locals[i]);

        return returned;
      };
    }

    if (["dFdx", "dFdy", "fwidth"].includes(name)) {
      const type = expressions[0] && this.expressionType(expressions[0], scope);
      if (args.length !== 1 || (type && !["float", "vec2", "vec3", "vec4"].includes(type)))
        throw new ShaderError(`${name} expects one float or floating-point vector`);
      this.usesDerivatives = true;
      const site = Symbol(name);

      return f => {
        if (!f.rt.context.derivatives)
          throw new ShaderError(`${name} requires a fragment quad`);

        return f.rt.context.derivatives.evaluate(site, name, args[0](f));
      };
    }

    if (["texture", "texture2D", "texture2DProj", "textureProj", "textureLod", "texture2DLod", "texture2DLodEXT",
      "textureGrad", "texture2DGradEXT"].includes(name)) {
      const explicitLod = name.includes("Lod"), explicitGrad = name.includes("Grad");
      const projected = name.endsWith("Proj");
      if (explicitLod ? args.length !== 3 : explicitGrad ? args.length !== 4 : args.length < 2 || args.length > 3)
        throw new ShaderError(`invalid arguments to ${name}`);
      if (!explicitLod && !explicitGrad) this.usesImplicitTextureSampling = true;
      const dxSite = Symbol("texture dx"), dySite = Symbol("texture dy");

      return f => {
        const sampler = args[0](f), coords = array(args[1](f)).map(num);
        const uv = projected ? [coords[0] / coords[coords.length - 1], coords[1] / coords[coords.length - 1]] : coords;
        const context = f.rt.context;
        if (explicitLod) return context.texture(sampler, uv, { lod: num(args[2](f)) });
        if (explicitGrad) return context.texture(sampler, uv,
          { dx: array(args[2](f)).map(num), dy: array(args[3](f)).map(num) });
        const bias = args[2] ? num(args[2](f)) : 0;

        if (context.derivatives && (context.textureNeedsFootprint?.(sampler) ?? true)) {
          const dx = context.derivatives.evaluate(dxSite, "dFdx", uv, true) as number[];
          const dy = context.derivatives.evaluate(dySite, "dFdy", uv, true) as number[];

          return context.texture(sampler, uv, { dx, dy, bias });
        }

        return context.texture(sampler, uv, { lod: 0, bias });
      };
    }

    const builtin = compileBuiltin(name, args);
    if (!builtin)
      throw new ShaderError(`unsupported function '${name}'`);

    return builtin;
  }

  run(inputs: Record<string, Value>, context: ShaderContext): Record<string, Value> | null {
    if (!this.main)
      throw new ShaderError("missing main()");
    const globals: Value[] = new Array(this.globalSlots.size);

    for (const name in inputs) {
      const value = context.derivatives ? copy(inputs[name]) : inputs[name];
      // Matrix uniforms arrive as column-major arrays in the transport protocol.
      globals[this.globalSlot(name)] = this.importValue(value, this.matrixInputTypes.get(name));
    }

    const rt: Runtime = { remaining: context.budget ?? 50000, depth: 0, returned: 0, context };
    const frame: Frame = { locals: [], globals, rt };

    try {
      // A global already provided from outside (uniform, varying or state from an earlier stage) is not redeclared.
      for (const declaration of this.declarations)
        if (globals[declaration.slot] === undefined)
          declaration.exec(frame);
      const flow = this.main.body({ locals: new Array(this.main.slots), globals, rt });
      if (flow === FLOW_DISCARD)
        return null;
      const result: Record<string, Value> = {};

      for (const [name, slot] of this.globalSlots) {
        const value = globals[slot];
        if (value !== undefined)
          result[name] = value;
      }

      return result;
    } catch(error) {
      if (error === DISCARD)
        return null;
      throw error;
    }
  }

  private needsMatrixImport(type: string): boolean {
    if (type.endsWith("[]")) return this.needsMatrixImport(type.slice(0, -2));

    return /^mat[234]$/.test(type) || !!this.structs.get(type)?.some(field => this.needsMatrixImport(field.type!));
  }

  private importValue(value: Value, type?: string): Value {
    if (!type) return value;
    if (type.endsWith("[]") && Array.isArray(value))
      return value.map(item => this.importValue(item, type.slice(0, -2)));
    if (/^mat[234]$/.test(type) && Array.isArray(value)) return matrixValue(value as number[]);
    const fields = this.structs.get(type);
    if (fields && value && typeof value === "object" && !Array.isArray(value))
      return Object.fromEntries(fields.map(field => [field.name!,
        this.importValue(value[field.name!], field.type! + (field.expressions?.[1] ? "[]" : ""))]));

    return value;
  }
}

/** Evaluates a preprocessor constant expression. */
function evaluateConstant(expr: Expr): Value {
  const program = new CompiledProgram(null);
  const rt: Runtime = { remaining: 50000, depth: 0, returned: 0, context: { texture: () => [0, 0, 0, 0] } };

  return program.compileExpr(expr, new CompileScope(null, { slots: 0 }))({ locals: [], globals: [], rt });
}

interface Macro { body: string; parameters?: string[] }

function expandMacros(source: string, macros: Map<string, Macro>, disabled = new Set<string>(), depth = 0,
  budget = { remaining: 100000 }): string {
  if (depth > 64 || --budget.remaining <= 0)
    throw new ShaderError("macro expansion budget exceeded");
  // Keep numeric tokens intact (notably exponent notation and hexadecimal literals).
  const tokens = source.match(/0[xX][\da-fA-F]+[uU]?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?[fFuU]?|[A-Za-z_]\w*|\s+|[^\s]/g) ?? [];
  let result = "";

  for (let i = 0; i < tokens.length; i++) {
    const name = tokens[i];
    let macro = macros.get(name);

    if (!macro || disabled.has(name)) {
      result += name; continue;
    }

    // An object macro may alias a function macro: #define APPLY SCALE.
    const aliases = new Set([name]);

    while (!macro.parameters && /^[A-Za-z_]\w*$/.test(macro.body)) {
      const alias = macro.body, next = macros.get(alias);
      if (!next || aliases.has(alias) || disabled.has(alias)) break;
      aliases.add(alias);
      macro = next;
    }

    let body = macro.body;

    if (macro.parameters) {
      let open = i + 1;
      while (/^\s+$/.test(tokens[open] ?? "")) open++;

      if (tokens[open] !== "(") {
        result += name; continue;
      }

      const actual: string[] = [];
      let nesting = 1, arg = "", j = open + 1;

      for (; j < tokens.length; j++) {
        const token = tokens[j];
        if (token === "(") nesting++;
        if (token === ")") nesting--;

        if (!nesting) {
          if (arg.trim() || actual.length || macro.parameters.length) actual.push(arg); break;
        }

        if (token === "," && nesting === 1) {
          actual.push(arg); arg = "";
        } else arg += token;
      }

      if (nesting) throw new ShaderError(`unterminated macro '${name}'`);
      if (actual.length !== macro.parameters.length) throw new ShaderError(`invalid macro arguments for '${name}'`);

      const replacements = new Map(macro.parameters.map((p, k) =>
        [p, expandMacros(actual[k], macros, disabled, depth + 1, budget)]));

      body = body.replace(/\b[A-Za-z_]\w*\b/g, word => replacements.get(word) ?? word);
      i = j;
    }

    result += expandMacros(body, macros, new Set([...disabled, ...aliases]), depth + 1, budget);
    if (result.length > 1000000) throw new ShaderError("macro expansion budget exceeded");
  }

  return result;
}

function preprocess(
  source: string,
  defines: Record<string, string | number | boolean>,
  includes: Record<string, string>,
  depth = 0,
  macros = new Map<string, Macro>(Object.entries(defines).map(([k, v]) =>
    [k, { body: String(v === true ? 1 : v === false ? 0 : v) }]))): string {
  if (depth > 32) throw new ShaderError("include depth exceeded");
  const conditions: { parent: boolean; matched: boolean; seenElse: boolean }[] = [];
  let active = true;

  const evaluate = (condition: string) => {
    const defined = condition.replace(/defined\s*(?:\(\s*(\w+)\s*\)|(\w+))/g,
      (_, a, b) => macros.has(a ?? b) ? "1" : "0");

    const tokens = tokenize(expandMacros(defined, macros)).map(token => /^[A-Za-z_]\w*$/.test(token) ? "0" : token);
    const parser = new Parser(tokens), expr = parser.expression();
    if (parser.peek()) throw new ShaderError("invalid preprocessor expression");

    return !!evaluateConstant(expr);
  };

  const out: string[] = [], pending: string[] = [];

  const flush = () => {
    if (pending.length) out.push(expandMacros(pending.splice(0).join("\n"), macros));
  };

  const clean = source.replace(/\\\r?\n/g, "").replace(/\/\*[\s\S]*?\*\//g,
    comment => comment.replace(/[^\n]/g, " ")).replace(/\/\/[^\n]*/g, "");

  for (const line of clean.split(/\r?\n/)) {
    const directive = line.match(/^\s*#\s*(\w+)\s*(.*)$/);

    if (!directive) {
      if (active) pending.push(line); continue;
    }

    flush();
    const [, command, arg] = directive;
    if (["if", "ifdef", "ifndef"].includes(command)) {
      const value: boolean = active && (command === "ifdef" ? macros.has(arg.trim())
        : command === "ifndef" ? !macros.has(arg.trim()) : evaluate(arg));

      conditions.push({ parent: active, matched: value, seenElse: false });
      active = value;
    } else if (command === "else" || command === "elif") {
      const condition = conditions[conditions.length - 1];
      if (!condition || condition.seenElse) throw new ShaderError("unmatched conditional");
      const value = condition.parent && !condition.matched && (command === "else" || evaluate(arg));
      condition.seenElse = command === "else";
      condition.matched ||= value;
      active = value;
    } else if (command === "endif") {
      const condition = conditions.pop();
      if (!condition) throw new ShaderError("unmatched #endif");
      active = condition.parent;
    } else if (active && command === "define") {
      const match = arg.match(/^(\w+)(?:\(([^)]*)\))?(.*)$/);
      if (!match || match[3].includes("#")) throw new ShaderError("invalid macro definition");
      const parameters = match[2] === undefined ? undefined : match[2].trim() ? match[2].split(",").map(p => p.trim()) : [];
      if (parameters && (parameters.some(p => !/^[A-Za-z_]\w*$/.test(p)) || new Set(parameters).size !== parameters.length))
        throw new ShaderError("invalid macro parameters");
      macros.set(match[1], { body: match[3].trim(), parameters });
    } else if (active && command === "undef") macros.delete(arg.trim());
    else if (active && command === "include") {
      const key = arg.match(/^<(\w+)>$/)?.[1];
      if (!key || !(key in includes)) throw new ShaderError(`unknown include ${arg}`);
      out.push(preprocess(includes[key], {}, includes, depth + 1, macros));
    } else if (active && command === "error") throw new ShaderError(`#error ${arg}`);
    else if (active && !["version", "extension", "pragma"].includes(command))
      throw new ShaderError(`unsupported directive #${command}`);
  }

  flush();
  if (conditions.length) throw new ShaderError("unterminated preprocessor conditional");

  return out.join("\n");
}

export class CpuShader {
  private program: CompiledProgram;
  readonly varyings: string[];
  readonly outputs: string[];

  constructor(
    source: string,
    defines: Record<string, string | number | boolean> = {},
    includes: Record<string, string> = {}) {
    const parsed = new Parser(tokenize(preprocess(source, defines, includes))).parse();
    this.program = new CompiledProgram(parsed);
    this.varyings = parsed.varyings;
    this.outputs = parsed.outputs;
  }

  get usesDerivatives(): boolean { return this.program.usesDerivatives; }

  get usesImplicitTextureSampling(): boolean { return this.program.usesImplicitTextureSampling; }

  /** Four fragment inputs in bottom-left, bottom-right, top-left, top-right order. */
  runQuad(inputs: Record<string, Value>[], context: ShaderContext): (Record<string, Value> | null)[] {
    if (inputs.length !== 4) throw new ShaderError("a fragment quad requires four inputs");
    if (!this.usesDerivatives && !this.usesImplicitTextureSampling)
      return inputs.map(input => this.run(input, context));

    return evaluateDerivativeQuad((lane, derivatives) => this.run(inputs[lane], { ...context, derivatives }),
      undefined, (context.budget ?? 50000) * 4);
  }

  run(inputs: Record<string, Value>, context: ShaderContext): Record<string, Value> | null {
    return this.program.run(inputs, context);
  }
}
