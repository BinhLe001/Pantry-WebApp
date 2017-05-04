var express = require('express');
var router = express.Router();

router.get('/', function (req, res, next) {
  res.render('welcome', { title: 'Express' });
});

router.get('/about', function (req,res,next) {
	console.log("about req");
	res.render('about');
});

router.get('/home', function (req,res,next) {
	res.render('home');
});

router.get('/saved', function (req,res,next) {
	res.render('saved');
});

router.get('/aboutwelcome', function (req,res,next) {
	res.render('aboutwelcome');
});
module.exports = router;
