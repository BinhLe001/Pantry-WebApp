var express = require('express');
var router = express.Router();
var http = require('http');
var request = require('request');

router.get('/', function(req,res,next) {
	res.render('home');
});

var db = require('../db-setup.js');

router.get('/recipe', function (req, res, next) {
	var id = process.env.YUMMLY_ID;
	var key = process.env.YUMMLY_KEY;
	var data = req.query.id_number;
	finalSearch = 'http://api.yummly.com/v1/api/recipe/';
	finalSearch += data;
	finalSearch += '?_app_id='
	finalSearch += id;
	finalSearch += '&_app_key=';
	finalSearch += key;
	var request = require('request');
	request(finalSearch, function (error, response, body) {
		if (!error && response.statusCode == 200) {
  			res.send(JSON.parse(body));
		} else {
			res.send('Error! Get Recipe Request returned no data');
		}
	});
});

var getNutritionString = function(nutrient, min, max) {
	var Max = max.toString();
	var Min = min.toString();
	if (nutrient === 'calorie') {
		if (min === 0) {
			toReturn = '&nutrition.ENERC_KCAL.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.ENERC_KCAL.min=' + Min;
		} else {
			toReturn = '&nutrition.ENERC_KCAL.min=' + Min + '&nutrition.ENERC_KCAL.max=' + Max;
		}
	} else if (nutrient === 'carb') {
		if (min === 0) {
			toReturn = '&nutrition.CHOCDF.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.CHOCDF.min=' + Min;
		} else {
			toReturn = '&nutrition.CHOCDF.min=' + Min + '&nutrition.CHOCDF.max=' + Max;
		}
	} else if (nutrient === 'protein') {
		if (min === 0) {
			toReturn = '&nutrition.PROCNT.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.PROCNT.min=' + Min;
		} else {
			toReturn = '&nutrition.PROCNT.min=' + Min + '&nutrition.PROCNT.max=' + Max;
		}
	} else if (nutrient === 'sodium') {
		if (min === 0) {
			toReturn = '&nutrition.NA.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.NA.min=' + Min;
		} else {
			toReturn = '&nutrition.NA.min=' + Min + '&nutrition.NA.max=' + Max;
		}
	} else if (nutrient === 'cholesterol') {
		if (min === 0) {
			toReturn = '&nutrition.CHOLE.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.CHOLE.min=' + Min;
		} else {
			toReturn = '&nutrition.CHOLE.min=' + Min + '&nutrition.CHOLE.max=' + Max;
		}
	} else if (nutrient === 'fat') {
		if (min === 0) {
			toReturn = '&nutrition.FAT.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.FAT.min=' + Min;
		} else {
			toReturn = '&nutrition.FAT.min=' + Min + '&nutrition.FAT.max=' + Max;
		}
	} else if (nutrient === 'sugar') {
		if (min === 0) {
			toReturn = '&nutrition.SUGAR.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.SUGAR.min=' + Min;
		} else {
			toReturn = '&nutrition.SUGAR.min=' + Min + '&nutrition.SUGAR.max=' + Max;
		}
	} else if (nutrient === 'fiber') {
		if (min === 0) {
			toReturn = '&nutrition.FIBTG.max=' + Max;
		} else if (max === 0) {
			toReturn = '&nutrition.FIBTG.min=' + Min;
		} else {
			toReturn = '&nutrition.FIBTG.min=' + Min + '&nutrition.FIBTG.max=' + Max;
		}
	}
	return toReturn;
};

var getNutrition = function(weight, period) {
	toReturn = '';
	if (period === '0') {
		if (weight === '0') { //Lose weight, Before workout
			toReturn += getNutritionString('sodium', 0, .4);
			toReturn += getNutritionString('cholesterol', 0, .10);
			toReturn += getNutritionString('fat', 0, 8);
			toReturn += getNutritionString('sugar', 0, 8);
			toReturn += getNutritionString('fiber', 4, 0);
			toReturn += getNutritionString('calorie', 100, 450);
			toReturn += getNutritionString('carb', 0, 40);
			toReturn += getNutritionString('protein', 9, 0);
		} else if (weight === '1') { //Maintain weight, Before workout
			toReturn += getNutritionString('calorie', 200, 600);
			toReturn += getNutritionString('carb', 20, 50);
			toReturn += getNutritionString('protein', 9, 40);
			toReturn += getNutritionString('fat', 0, 10);
			toReturn += getNutritionString('sodium', 0, .5);
			toReturn += getNutritionString('cholesterol', 0, .15);
			toReturn += getNutritionString('sugar', 0, 13);
			toReturn += getNutritionString('fiber', 4, 0);
		} else { //Gain weight, Before workout
			toReturn += getNutritionString('sodium', 0, .5);
			toReturn += getNutritionString('cholesterol', 0, .10);
			toReturn += getNutritionString('fat', 0, 12);
			toReturn += getNutritionString('sugar', 0, 10);
			toReturn += getNutritionString('fiber', 6, 0);
			toReturn += getNutritionString('calorie', 400, 750);
			toReturn += getNutritionString('carb', 35, 0);
			toReturn += getNutritionString('protein', 17, 0);
		}
	} else {
		if (weight === '0') { //Lose weight, After workout
			toReturn += getNutritionString('calorie', 300, 600);
			toReturn += getNutritionString('carb', 0, 60);
			toReturn += getNutritionString('protein', 12, 0);
			toReturn += getNutritionString('sodium', 0, .5);
			toReturn += getNutritionString('cholesterol', 0, .10);
			toReturn += getNutritionString('sugar', 0, 10);
			toReturn += getNutritionString('fiber', 5, 0);
			toReturn += getNutritionString('fat', 0, 10);
		} else if (weight === '1') { //Maintain weight, After workout
			toReturn += getNutritionString('calorie', 350, 700);
			toReturn += getNutritionString('carb', 25, 70);
			toReturn += getNutritionString('protein', 13, 50);
			toReturn += getNutritionString('sodium', 0, .6);
			toReturn += getNutritionString('cholesterol', 0, .10);
			toReturn += getNutritionString('sugar', 0, 12);
			toReturn += getNutritionString('fiber', 3, 0);
			toReturn += getNutritionString('fat', 0, 11);
		} else { //Gain weight, after workout
			toReturn += getNutritionString('calorie', 575, 0);
			toReturn += getNutritionString('carb', 55, 0);
			toReturn += getNutritionString('protein', 25, 0);
			toReturn += getNutritionString('sodium', 0, .7);
			toReturn += getNutritionString('cholesterol', 0, .10);
			toReturn += getNutritionString('sugar', 0, 10);
			toReturn += getNutritionString('fiber', 6, 0);
			toReturn += getNutritionString('fat', 0, 12);
		}
	}
	return toReturn;
};

var allergy_to_code = function(allergy) {
	if (allergy === 'dairy') return 'allowedAllergy[]=396^Dairy-Free';
	if (allergy === 'egg') return 'allowedAllergy[]=397^Egg-Free';
	if (allergy === 'gluten') return 'allowedAllergy[]=393^Gluten-Free';
	if (allergy === 'peanut') return 'allowedAllergy[]=394^Peanut-Free';
	if (allergy === 'seafood') return 'allowedAllergy[]=398^Seafood-Free';
	if (allergy === 'sesame') return 'allowedAllergy[]=399^Sesame-Free';
	if (allergy === 'soy') return 'allowedAllergy[]=400^Soy-Free';
	if (allergy === 'sulfite') return 'allowedAllergy[]=401^Sulfite-Free';
	if (allergy === 'wheat') return 'allowedAllergy[]=392^Wheat-Free';
	if (allergy === 'vegetarian') return 'allowedDiet[]=387^Lacto-ovo%20vegetarian';
	if (allergy === 'pescetarian') return 'allowedDiet[]=390^Pescetarian';
	if (allergy === 'vegan') return 'allowedDiet[]=386^Vegan';
	if (allergy === 'paleo') return 'allowedDiet[]=403^Paleo';
	return '';
};

router.get('/search', function (req, res, next) {
	var id = process.env.YUMMLY_ID;
	var key = process.env.YUMMLY_KEY;
	var data = req.query.search;
	var finalSearch = '';
	if (data[0][0] !== '0') {
		finalSearch += 'q=';
		finalSearch += encodeURIComponent(data[0][0]);
		finalSearch += '&';
	}
	if (data[1][0] !== '0') {
		for (i=0; i<data[1].length; i++) {
			finalSearch += "allowedIngredient[]=";
			finalSearch += encodeURIComponent(data[1][i]);
			finalSearch += '&';
		}
	}
	if (data[2][0] !== '0') {
		for (i=0; i<data[2].length; i++) {
			finalSearch += allergy_to_code(data[2][i]);
			finalSearch += '&';
		}
	}
	if (data[3][0] !== '0') {
		for (i=0; i<data[3].length; i++) {
			finalSearch += "allowedCuisine[]=cuisine^cuisine-";
			finalSearch += data[3][i].toLowerCase();
			finalSearch += '&';
		}
	}
    if (finalSearch.substring(finalSearch.length-1) == "&")
    {
        finalSearch = finalSearch.substring(0, finalSearch.length-1);
    }

    if (data[6][0] === '0' || data[6][0] === '1' || data[6][0] === '2') {
    	if (data[7][0] === '0' || data[7][0] === '1') {
    		finalSearch += getNutrition(data[6][0], data[7][0]);
    	}
    } 

    finalSearch += '&requirePictures=true';
    finalSearch += '&maxTotalTimeInSeconds=';
    var seconds = data[4][0]*60;
    finalSearch += seconds.toString();
    finalSearch += '&maxResult=';
    var page = data[5][0];
    finalSearch += page.toString();
    finalSearch += '&start=';
    finalSearch += page*data[5][1].toString();
	var finalURL = 'http://api.yummly.com/v1/api/recipes?_app_id='+id+'&_app_key='+key+'&'+finalSearch;
	var request = require('request');
	request(finalURL, function (error, response, body) {
		if (!error && response.statusCode == 200) {
  			var recipes = JSON.parse(body).matches;
  			var info = [];
  			for (i=0; i<recipes.length; i++) {
  				var newRecipe = {};
  				newRecipe.url = 'http://www.yummly.com/recipe/' + recipes[i].id;
  				newRecipe.imageurl = recipes[i].smallImageUrls[0];
  				newRecipe.ingredients = recipes[i].ingredients;
  				newRecipe.recipeName = recipes[i].recipeName;
  				newRecipe.rating = recipes[i].rating;
  				newRecipe.timeToMake = recipes[i].totalTimeInSeconds;
  				newRecipe.id = recipes[i].id;
  				info.push(newRecipe);
			}
			res.send(info);
		} else {
			console.log(error);
			res.send('');
		}
	});
});

router.get('/recipeinput', function (req, res, next) {
	var ingredient = req.query.search;
	res.send(ingredient);
});

router.get('/ingredientinput', function (req, res, next) {
	var ingredient = req.query.search;
	res.send(ingredient);
});

router.get('/restrictioninput', function (req, res, next) {
	var restriction = req.query.restrictions;
	res.send(restriction);
});

router.get('/timeinput', function (req, res, next) {
	var time = req.query.times;
	res.send(time);
});

router.get('/cuisineinput', function (req, res, next) {
	var cuisine = req.query.cuisines;
	res.send(cuisine);
});

router.post('/togglesaved', function (req, res, next) {
	var data = req.body.array;
	data = JSON.parse(data);
	db.users.find({userid:data[2]}).toArray( function (err, peeps) {
		var recipeSaved = "0";
		if (peeps[0].saved.indexOf(data[1]) === -1) {
			recipeSaved = "0";
		} else {
			recipeSaved = "1";
		}
		if (recipeSaved === '0') {
			if (peeps[0].saved.length < 36) {
				db.users.update(
					{userid: data[2]},
					{$addToSet:
						{saved: data[1]}
					}
				);
			} else {
				recipeSaved = '2';
			}
		} else {
			db.users.update(
				{userid: data[2]},
				{$pull:
					{saved: data[1]}
				}
			);
		}
		res.send(recipeSaved);
	});
});

router.get('/display', function (req, res, next) {
	var data = req.query.array;
	data = JSON.parse(data);
	db.users.find({userid:data[1]}).toArray(function (err, person) {
		var settings = {};
		var new_user = undefined;
		if (person.length === 0) {
			db.users.insert({userid:data[1],username:data[0],saved:[],restrictions:[],cuisines:[],ingredients:[],recipes: "",cooktime: "120",goal: "",state: ""});
			settings.restrictions = [];
			settings.cuisines = [];
			settings.ingredients = [];
			settings.recipes = "";
			settings.cooktime = "120";
			settings.goal = "";
			settings.state = "";
			console.log('inserted user');
			var new_user = true;
		} else {
			settings.restrictions = person[0].restrictions;
			settings.cuisines = person[0].cuisines;
			settings.ingredients = person[0].ingredients;
			settings.recipes = person[0].recipes;
			settings.cooktime = person[0].cooktime;
			settings.goal = person[0].goal;
			settings.state = person[0].state;
			var new_user = false;
		}
		res.send([settings,new_user]);
	});
});

router.post('/updatesetting', function (req, res, next) {
	var data = req.body.array;
	data = JSON.parse(data);
	if (data[2] === 'recipe') {
		console.log('recipe');
		db.users.update(
			{userid: data[0]},
			{$set:
				{recipes:data[1]}
			}
		);
	} else if (data[2] === 'cooktime') {
		console.log('update cooktime:');
		console.log(data[1]);
		db.users.update(
			{userid: data[0]},
			{$set:
				{cooktime:data[1]}
			}
		);
	} else if (data[2] === 'goal') {
		db.users.update(
			{userid: data[0]},
			{$set:
				{goal:data[1]}
			}
		);
	} else {
		db.users.update(
			{userid: data[0]},
			{$set:
				{state:data[1]}
			}
		);
	}
	res.send('updated setting');
});

router.post('/savesetting', function (req, res, next) {
	var data = req.body.array;
	data = JSON.parse(data);
	if (data[2] === 'restriction') {
		db.users.update(
			{userid: data[0]},
			{$addToSet:
				{restrictions: data[1]}
			}
		);
	} else if (data[2] === 'cuisine') {
		db.users.update(
			{userid: data[0]},
			{$addToSet:
				{cuisines: data[1]}
			}
		);
	} else {
		db.users.update(
			{userid: data[0]},
			{$addToSet:
				{ingredients: data[1]}
			}
		);
	}
	res.send('setting saved');
});

router.post('/removesetting', function (req, res, next) {
	var data = req.body.array;
	data = JSON.parse(data);
	if (data[2] === 'restriction') {
		db.users.update(
			{userid: data[0]},
			{$pull:
				{restrictions: data[1]}
			}
		);	
	} else if (data[2] === 'cuisine') {
		db.users.update(
			{userid: data[0]},
			{$pull:
				{cuisines: data[1]}
			}
		);
	} else  {
		db.users.update(
			{userid: data[0]},
			{$pull:
				{ingredients: data[1]}
			}
		);
	}
	res.send('setting removed');
});

router.post('/removeallingredients', function (req, res, next) {
	var data = req.body.array;
	var data = JSON.parse(data);
	db.users.update(
		{userid: data[0]},
		{$set:
			{ingredients: []}
		}
	);
	res.send('remove all ingredients');
});


router.get('/isrecipesaved', function (req, res, next) {
	var data = req.query.array;
	data = JSON.parse(data);
	db.users.find({userid:data[2]}).toArray( function (err, peeps) {
		if (peeps.length === 0) {
			recipeSaved = "0";
		} else {
			var recipeSaved = "0";
			if (peeps[0].saved.indexOf(data[1]) === -1) {
				recipeSaved = "0";
			} else {
				recipeSaved = "1";
			}
		}
		res.send(recipeSaved);
	});
});

module.exports = router;