var mongo = require('mongodb').MongoClient;
var dbConnectionUrl = (process.env.MONGOLAB_URI ||'mongodb://localhost/db-data');
var collections = {};

mongo.connect(dbConnectionUrl, function (err, db) {
  if (err) {
    console.log('Can not connect to MongoDB. This sucks');
    console.log(err.message);
    return;
  }
  collections.users = db.collection('users');
});

module.exports = collections;