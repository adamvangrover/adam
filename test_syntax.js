const fs = require('fs');
try {
  require('./showcase/js/comprehensive_credit_dashboard.js');
  console.log('Syntax OK');
} catch (e) {
  if (e instanceof ReferenceError || e instanceof TypeError) {
    console.log('Syntax OK (Hit runtime error)');
  } else {
    console.log(e);
  }
}
