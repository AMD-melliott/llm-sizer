// Jest mock for chalk (ESM-only package)
// Returns strings unmodified — chalk formatting is irrelevant in unit tests.

function identity(str) {
  return String(str ?? '');
}

const colorFn = (str) => identity(str);

// Each color/style property is itself callable and has chainable sub-properties.
function makeChalk() {
  const handler = {
    apply(_target, _thisArg, args) {
      return identity(args[0]);
    },
    get(_target, prop) {
      // bold, dim, red, green, etc. — return a new proxy so chaining works
      return new Proxy(colorFn, handler);
    },
  };
  return new Proxy(colorFn, handler);
}

const chalk = makeChalk();

module.exports = chalk;
module.exports.default = chalk;
