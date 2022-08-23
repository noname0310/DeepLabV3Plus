module.exports = {
    "env": {
        "browser": true,
        "es2021": true
    },
    "extends": [
        "eslint:recommended",
        "plugin:react/recommended",
        "plugin:@typescript-eslint/recommended"
    ],
    "parser": "@typescript-eslint/parser",
    "parserOptions": {
        "ecmaFeatures": {
            "jsx": true
        },
        "ecmaVersion": "latest",
        "sourceType": "module",
        "project": "./tsconfig.json"
    },
    "plugins": [
        "react",
        "@typescript-eslint",
        "simple-import-sort"
    ],
    "rules": {
        "react/react-in-jsx-scope": "off",
        "@typescript-eslint/no-non-null-assertion": "off",
        "@typescript-eslint/no-explicit-any": "off",
        "@typescript-eslint/no-unused-vars": "off",
        "@typescript-eslint/explicit-member-accessibility": [
            "error",
            {
                "accessibility": "explicit",
                "overrides": {
                    "accessors": "explicit",
                    "constructors": "explicit",
                    "methods": "explicit",
                    "properties": "explicit",
                    "parameterProperties": "explicit"
                }
            }
        ],
        "@typescript-eslint/prefer-readonly": ["error"],
        "@typescript-eslint/explicit-function-return-type": ["error"],
        "@typescript-eslint/array-type": ["error"],
        "@typescript-eslint/brace-style": [
            "error",
            "1tbs"
        ],
        "@typescript-eslint/prefer-includes": ["error"],
        "@typescript-eslint/space-before-blocks": ["error"],
        "@typescript-eslint/type-annotation-spacing": [
            "error",
            {
                "before": false,
                "after": true,
                "overrides": {
                    "arrow": {
                        "before": true,
                        "after": true
                    }
                }
            }
        ],
        "no-debugger": "warn",
        "indent": [
            "error",
            4
        ],
        "linebreak-style": [
            "error",
            "unix"
        ],
        "quotes": [
            "error",
            "double"
        ],
        "semi": [
            "error",
            "always"
        ],
        "comma-dangle": [
            "error",
            "never"
        ],
        "comma-spacing": [
            "error"
        ],
        "simple-import-sort/imports": "error",
        "simple-import-sort/exports": "error"
    },
    "settings": {
        "react": {
            "version": "detect"
        }
    }
};
