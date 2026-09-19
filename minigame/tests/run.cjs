'use strict';
const suites = [require('./game.test.cjs'), require('./ui.test.cjs')];
let count = 0;
for (const suite of suites) {
  for (const name of suite()) { console.log(`PASS ${name}`); count++; }
}
console.log(`\n${count} tests passed. No browser pixel or audio-output assertions.`);
