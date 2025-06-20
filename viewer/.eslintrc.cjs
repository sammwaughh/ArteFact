/** @type {import('eslint').Linter.Config} */
module.exports = {
  root: true,

  parser: '@typescript-eslint/parser',
  parserOptions: { project: './tsconfig.eslint.json' },

  plugins: [
    '@typescript-eslint',
    'react',
    'react-hooks',
    'import'            // import-plugin rules
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
    // modern React: no need to put React in scope
    'react/react-in-jsx-scope': 'off',

    // loosen target="_blank" warning for early prototypes
    'react/jsx-no-target-blank': 'off'
  },

  // 👉 NEW: allow devDependencies in build-config files
  overrides: [
    {
      files: ['vite.config.*'],
      rules: {
        'import/no-extraneous-dependencies': [
          'error',
          { devDependencies: true }
        ]
      }
    }
  ]
};
