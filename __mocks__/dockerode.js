module.exports = class Docker {
  constructor() {}
  listContainers() { return Promise.resolve([]); }
  getContainer() { return { inspect: () => Promise.resolve({}) }; }
};
