/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  parserOptions: { project: './tsconfig.eslint.json' },

  plugins: [
    '@typescript-eslint',
    'react',
    'react-hooks',
    'import'          // <<-- loads the import rules
  ],

  extends: [
    'airbnb-typescript/base',
    'airbnb-typescript',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended',
    'plugin:react-hooks/recommended',
    'plugin:import/recommended',
    'plugin:import/typescript'
  ],

  settings: {
    react: { version: 'detect' }
  },

  rules: {
    // modern React: no need to import React in scope
    'react/react-in-jsx-scope': 'off',

    // target="_blank" rule is too strict for MVP; re-enable later if desired
    'react/jsx-no-target-blank': 'off'
  }
};
