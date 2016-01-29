var express = require('express');
var router = express.Router();
var db = require('../db-setup.js');

router.get('/', function (req,res,next) {
	res.render('saved');
});

router.get('/display', function (req, res, next) {
	var data = req.query.array;
	data = JSON.parse(data);
	db.users.find({userid:data[1]}).toArray(function (err, person) {
		if (person.length === 0) {
			console.log('new user');
			db.users.insert({userid:data[1],username:data[0],saved:[],restrictions:[],cuisines:[],ingredients:[],recipes: "", cooktime: "120", goal: "", state: ""});
			res.send([]);
		} else {
			res.send(person[0].saved);
		}
	});
});

router.post('/remove', function (req, res, next) {
	var data = req.body.array;
	data = JSON.parse(data);
	db.users.update(
		{userid: data[2]},
		{$pull:
			{saved: data[1]}
		}
	);
	res.send('removed recipe');
});

module.exports = router;
