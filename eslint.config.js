import js from "@eslint/js";
import stylistic from "@stylistic/eslint-plugin";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  {
    ignores: ["**/dist/**", "**/node_modules/**"],
  },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["**/*.{ts,js}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.browser, ...globals.worker },
    },
    plugins: {
      "@stylistic": stylistic,
    },
    rules: {
      "@typescript-eslint/no-unused-vars": ["warn", {
        argsIgnorePattern: "^_",
        varsIgnorePattern: "^_",
        caughtErrorsIgnorePattern: "^_",
      }],
      "@typescript-eslint/no-explicit-any": "warn",
      "@typescript-eslint/no-inferrable-types": "off",
      "@typescript-eslint/consistent-type-imports": ["error", { fixStyle: "inline-type-imports" }],

      // Only warn and error are legitimate logging in a library.
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "no-debugger": "error",
      "prefer-const": "error",
      "no-var": "error",
      "eqeqeq": ["error", "always", { null: "ignore" }],
      "no-duplicate-imports": ["error", { allowSeparateTypeImports: true }],

      "@stylistic/indent": ["error", 2, { SwitchCase: 1 }],
      "@stylistic/quotes": ["error", "double", { avoidEscape: true }],
      "@stylistic/semi": ["error", "always"],
      "@stylistic/comma-dangle": ["error", "always-multiline"],
      "@stylistic/no-trailing-spaces": "error",
      "@stylistic/eol-last": ["error", "always"],
      "@stylistic/object-curly-spacing": ["error", "always"],
      "@stylistic/array-bracket-spacing": ["error", "never"],
      "@stylistic/key-spacing": ["error", { beforeColon: false, afterColon: true }],
      "@stylistic/space-before-function-paren": ["error", "never"],
      "@stylistic/space-infix-ops": "error",
      "@stylistic/keyword-spacing": "error",
      "@stylistic/brace-style": ["error", "1tbs", { allowSingleLine: true }],
      "@stylistic/max-len": ["warn", {
        code: 120,
        ignoreUrls: true,
        ignoreStrings: true,
        ignoreTemplateLiterals: true,
        ignoreRegExpLiterals: true,
        ignoreComments: true,
      }],

      // Blank lines around declarations: functions, classes, types and multi-line blocks stand apart.
      "@stylistic/lines-between-class-members": ["error", {
        enforce: [
          { blankLine: "always", prev: "*", next: "method" },
          { blankLine: "always", prev: "method", next: "*" },
        ],
      }],
      "@stylistic/padding-line-between-statements": ["error",
        { blankLine: "always", prev: "import", next: "*" },
        { blankLine: "any", prev: "import", next: "import" },
        {
          blankLine: "always",
          prev: "*",
          next: ["function", "class", "interface", "type", "multiline-export", "multiline-const", "multiline-let", "block-like"],
        },
        {
          blankLine: "always",
          prev: ["function", "class", "interface", "type", "multiline-export", "multiline-const", "multiline-let", "block-like"],
          next: "*",
        },
        { blankLine: "always", prev: "*", next: "return" },
        // Grouped case labels (an empty case falling into the next) stay together.
        { blankLine: "any", prev: ["case", "default"], next: ["case", "default"] },
      ],
      "@stylistic/no-multiple-empty-lines": ["error", { max: 1, maxBOF: 0, maxEOF: 0 }],
    },
  },
  {
    // Command line tools and tests report through the console.
    files: ["**/scripts/**", "**/*.test.ts", "**/vite.config.ts"],
    languageOptions: { globals: globals.node },
    rules: { "no-console": "off" },
  },
);
