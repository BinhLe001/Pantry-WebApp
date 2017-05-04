$(document).ready(function(){

    var listToArray = function(list) {
        var array = [];
        list.each(function() { array.push($(this).text()) });
        return array;
    };

    var saveSetting = function(input,setting) {
        var user_id = localStorage.getItem("user_id");
        if (user_id !== null) {
            var thing = [user_id,input,setting];
            $.ajax({
                url: "/home/savesetting",
                type: "Post",
                data: {
                    array: JSON.stringify(thing)
                },
                success: function(data) {
                },
                error: function(xhr, status, error) {
                    console.log('Failed save setting')
                }
            });
        }
        return false;
    };

    var removeSetting = function(input,setting) {
        var user_id = localStorage.getItem("user_id");
        if (user_id !== null) {
            var thing = [user_id,input,setting];
            $.ajax({
                url: "/home/removesetting",
                type: "Post",
                data: {
                    array: JSON.stringify(thing)
                },
                success: function(data) {
                },
                error: function(xhr, status, error) {
                    console.log('Failed to remove setting')
                }
            });
        }
        return false;
    };

    var removeAllIngredients = function() {
        var user = localStorage.getItem("user");
        var user_id = localStorage.getItem("user_id");
        if (user_id !== null) {
            var thing = [user_id,user];
            $.ajax({
                url: "/home/removeallingredients",
                type: "Post",
                data: {
                    array: JSON.stringify(thing)
                },
                success: function(data) {
                },
                error: function(xhr, status, error) {
                    console.log('Failed to remove setting')
                }
            });
        }
        return false;
    };

    var updateSetting = function(input, setting) {
        var user_id = localStorage.getItem("user_id");
        if (user_id !== null) {
            var thing = [user_id,input,setting];
            $.ajax({
                url: "/home/updatesetting",
                type: "Post",
                data: {
                    array: JSON.stringify(thing)
                },
                success: function(data) {
                },
                error: function(xhr, status, error) {
                    console.log('Failed to remove setting');
                }
            });
        }
        return false;
    };

    var getTopTwoFlavors = function(flavors) {
        var maxScore = 0;
        var maxKey = null;
        Object.keys(flavors).forEach(function(key,index) {
            if (flavors[key] > maxScore) {
                maxScore = flavors[key];
                maxKey = key;
            }
        });
        var secondScore = 0;
        var secondKey = null;
        Object.keys(flavors).forEach(function(key,index) {
            if (key !== maxKey && flavors[key] > secondScore) {
                secondScore = flavors[key];
                secondKey = key;
            }
        });
        return [maxKey, secondKey];
    };

    var flavorToURL = function(flavor) {
        toReturn = undefined;
        if (flavor === "Bitter") {
            toReturn = "/images/icon/bitter.jpg";
        } else if (flavor === "Sour") {
            toReturn = "/images/icon/sour.png";
        } else if (flavor === "Salty") {
            toReturn = "/images/icon/salt.png";
        } else if (flavor === "Sweet") {
            toReturn = "/images/icon/sweet.ico";
        } else if (flavor === "Spicy") {
            toReturn = "/images/icon/Spicy.png";
        } else {
            toReturn = "/images/icon/savory_cropped.png";
        }
        return toReturn;
    };

    var addSearchParameter = function(list, new_parameter) {
        var innerIcon = $("<i/>", {
            class: 'remove icon'
        });
        var inner = $("<button/>", {
            class: 'remove-input ui icon button small'
        });
        var middle = $('<div/>', {
            class: 'current-selected-container',
            text: new_parameter
        });
        var outer = $('<li/>', {
            class: 'current-selected-element'
        });
        inner = innerIcon.wrap(inner).parent();
        middle = inner.wrap(middle).parent();                    
        outer = middle.wrap(outer).parent();
        list.append(outer);
    };

    var showNoParameterMessage = function() {
        var array = $("#rightpart > div");
        $(array[1]).show();
    };

    var hideNoParameterMessage = function() {
        var array = $("#rightpart > div");
        $(array[1]).hide();
    };

    var removeSearchResults = function() {
        var array = $("#rightpart > div");
        $(array[0]).empty();
    };

    var inSearchParameters = function(list, new_parameter) {
        thing = false;
        list.children().children().each(function() {
            if (new_parameter === $(this)[0].innerText) {
                thing = true;
            }
        });
        return thing;
    };

    var fitnessButtonSetting = function(but) {
        if (but === 'lose-weight') {
            $("#lose-weight").addClass("active");
        } else if (but === 'maintain-weight') {
            $("#maintain-weight").addClass("active");
        } else if (but === 'gain-weight') {
            $("#gain-weight").addClass("active");
        } else if (but === 'before-workout') {
            $("#before-workout").addClass("active");
        } else if (but === 'after-workout') {
            $("#after-workout").addClass("active");
        } else {
        }
    };

    var bothOrNoneChosen = function() {
        toReturn = false;
        if($('#weight-buttons').find('.active').length == 0) {
            if($('#state-buttons').find('.active').length == 0) {
                toReturn = true;
            } else {
                toReturn = false;
            }
        } else {
            if($('#state-buttons').find('.active').length == 0) {
                toReturn = false;
            } else {
                toReturn = true;
            }
        }
        return toReturn;
    };

    var noneChosen = function() {
        toReturn = false;
        if ($("#lose-weight").hasClass("active") || $("#maintain-weight").hasClass("active") || $("#gain-weight").hasClass("active")){
            toReturn = false;
        } else {
            if ($("#before-workout").hasClass("active") || $("#after-workout").hasClass("active")) {
                toReturn = false;
            }
            else {
                toReturn = true;
            }
        }
        return toReturn;
    };

    var allParametersEmpty = function() {
        var recipe = ($("#recipe-list").children().length === 0);
        var ingredients = ($("#ingredient-list").children().length === 0);
        var restrictions = ($("#restriction-list").children().length === 0);
        var cuisines = ($("#cuisine-list").children().length === 0);
        return (recipe && ingredients && restrictions && cuisines);
    };

    var showNoResultsMessage = function() {
        var array = $("#rightpart > div");
        $(array[2]).show();
    };

    var hideNoResultsMessage = function() {
        var array = $("#rightpart > div");
        $(array[2]).hide();
    };

    var showNeedBothMessage = function() {
        var array = $("#rightpart > div");
        $(array[3]).show();
    };

    var hideNeedBothMessage = function() {
        var array = $("#rightpart > div");
        $(array[3]).hide();
    };

    var getRecipeDetails = function(id_string, card_div) {
        $.ajax({
            url: "/home/recipe",
            type: "Get",
            data: {
                id_number: id_string
            },
            success: function(info) {
                // card_div is a div of class="maintainsize", with
                // info is the JSON object returned by the Get Recipe Response: 
                // https://developer.yummly.com/documentation#recipe
                card_div.find("a").attr('href', info.attribution.url);
                var original_url = info.images[0].hostedLargeUrl;
                if (original_url.slice(0,5) !== 'https') {
                    var original_url = original_url.slice(0,4) + 's' + original_url.slice(4);
                }
                card_div.find(".lazy").attr('src',original_url);
                var name = info.name;
                if (name.length > 29) {
                    name = name.substring(0,29);
                    name = name.trim();
                    name += '...';
                }
                card_div.find("h").text(name);
                var Recipe = id_string;
                var user = localStorage.getItem("user");
                var user_id = localStorage.getItem("user_id");
                var thing = [user,Recipe,user_id];
                $.ajax({
                    url: "/home/isrecipesaved",
                    type: "Get",
                    data: {
                        array: JSON.stringify(thing)
                    },
                    success: function(data) {
                        var saveButton = card_div.find(".save");
                        if (data === '0') {
                            saveButton.attr('title','Save recipe');
                        } else {
                            saveButton.addClass("green");
                            saveButton.attr('title','Remove recipe');
                        }
                    },
                    error: function(xhr, status, error) {
                        console.log('failed to toggle recipe save')
                    }
                });
                f1 = card_div.find("#flavor1");
                f2 = card_div.find("#flavor2");
                if (!jQuery.isEmptyObject(info.flavors)) {
                    topTwo = getTopTwoFlavors(info.flavors);
                    if (topTwo[0] === "Piquant") {
                        topTwo[0] = "Spicy";
                    }
                    if (topTwo[0] === "Meaty") {
                        topTwo[0] = "Savory";
                    }
                    if (topTwo[1] === "Piquant") {
                        topTwo[1] = "Spicy";
                    }
                    if (topTwo[1] === "Meaty") {
                        topTwo[1] = "Savory";
                    }
                    f1.attr('src',flavorToURL(topTwo[0]));
                    f1.attr('title',topTwo[0]);
                    f2.attr('src',flavorToURL(topTwo[1]));
                    f2.attr('title',topTwo[1]);
                } else {
                    f1.attr('src',"/images/icon/white_background.png");
                    f2.attr('src',"/images/icon/white_background.png");
                }
            },
            error: function(xhr, status, error) {
                console.log("Get Recipe failed");
            }
        });
    };

    var getSearchResults = function (num_results, page, load_more) {
        num_results = typeof num_results !== 'undefined' ? num_results : 24;
        page = typeof page !== 'undefined' ? page : 0;
        load_more = typeof load_more !== 'undefined' ? load_more: false;
        var recipe = listToArray($("#recipe-list li"));
        var ingredient = listToArray($("#ingredient-list li"));
        var restriction = listToArray($("#restriction-list li"));
        var cuisine = listToArray($("#cuisine-list li"));
        var time = $('#time-bar').val();
        var weight = $("#weight-buttons").find(".active");
        var period = $("#state-buttons").find(".active");
        time = parseInt(time);
        if (recipe.length === 0) {
            recipe = [0];
        }
        if (ingredient.length === 0) {
            ingredient = [0];
        }
        if (restriction.length === 0) {
            restriction = [0];
        }
        if (cuisine.length === 0) {
            cuisine = [0];
        }
        if (weight.length === 0) {
            weight = [3];
        } else if (weight.is("#lose-weight")) {
            weight = [0];
        } else if (weight.is("#maintain-weight")) {
            weight = [1];
        } else {
            weight = [2];
        }
        if (period.length === 0) {
            period = [2];
        } else if (period.is("#before-workout")) {
            period = [0];
        } else {
            period = [1];
        }
        var Search = [recipe,ingredient,restriction,cuisine,[time],[num_results,page,load_more],weight,period];
        $.ajax({
            url: "/home/search",
            type: "Get",
            data: {
                search: Search
            },
            success: function(recipes) {
                if (!load_more) {
                    removeSearchResults();
                    if (recipes.length === 0) {
                        showNoResultsMessage();
                    }
                }
                for (var i=0; i<recipes.length; i++) {
                    var level4link = $("<a/>", {
                        target: "_blank"
                    });
                    var level5img = $("<img/>", {
                        class:"lazy",
                    });
                    var level3linkdiv = $("<div/>", {
                        class:"image"
                    });
                    var level4span = $("<span/>", {
                        class:" floated like button-container",
                        title: recipes[i].id
                    });
                    var level4span2 = $("<span/>", {
                        class: "floated like flavors"
                    });
                    var level5button = $("<button/>", {
                        class:"ui icon button save small"
                    });
                    var level6i = $("<i/>", {
                        class:"plus icon"
                    });
                    var level5flav = $("<img/>", {
                        class:"flavor",
                        id:"flavor1"
                    });
                    var level5flav2 = $("<img/>", {
                        class:"flavor",
                        id:"flavor2"
                    });
                    var level4header = $("<h/>", {
                        class:"recipe-title"
                    });
                    var level3textdiv = $("<div/>", {
                        class:"content"
                    });
                    var level2div = $("<div/>", {
                        class:"ui fluid card"
                    });
                    var level1div = $("<div/>", {
                        class:"maintainsize"
                    });
                    level5button.append(level6i);
                    level4span.append(level5button);
                    level4span2.append(level5flav);
                    level4span2.append(level5flav2);
                    level3textdiv.append(level4header);
                    level3textdiv.append(level4span2);
                    level3textdiv.append(level4span);
                    level4link.append(level5img);
                    level3linkdiv.append(level4link);
                    level2div.append(level3linkdiv);
                    level2div.append(level3textdiv);
                    level1div.append(level2div);
                    $('.ui.three.grid').append(level1div);
                    getRecipeDetails(recipes[i].id,level1div);
                }
            },
            error: function(xhr, status, error) {
                console.log("Recipes failed to load");
            }
        });
    };

    var check = function() {
        if (localStorage.getItem("user")!==null) {
            var user = localStorage.getItem("user");
            var user_id = localStorage.getItem("user_id");
            var thing = [user, user_id];
            $.ajax({
                url: "/home/display",
                type: "Get",
                data: {
                    array: JSON.stringify(thing)
                },
                beforeSend: function() {
                    hideNoParameterMessage();
                    $("#home-loading").show();
                },
                success: function(display_data) {
                    {
                        $("#home-loading").hide();
                        var data = display_data[0];
                        var new_user = display_data[1];
                        var restrictions = data.restrictions;
                        var cuisines = data.cuisines;
                        var ingredients = data.ingredients;
                        var recipes = data.recipes;
                        var cooktime = data.cooktime;
                        var goal = data.goal;
                        var state = data.state;
                        if (new_user) {
                            location.reload(true);
                        } else if (restrictions.length > 0 || 
                            cuisines.length > 0 || 
                            ingredients.length > 0 || 
                            recipes.length > 0 ||
                            cooktime !== '120' ||
                            goal.length > 0 ||
                            state.length > 0) {
                                for (var z=0; z<restrictions.length; z++) {
                                    addSearchParameter($('#restriction-list'),restrictions[z]);
                                }
                                for (var x=0; x<cuisines.length; x++) {
                                    addSearchParameter($('#cuisine-list'),cuisines[x]);
                                }
                                for (var c=0; c<ingredients.length; c++) {
                                    addSearchParameter($('#ingredient-list'),ingredients[c]);
                                }
                                if (recipes.length > 0) {
                                    addSearchParameter($('#recipe-list'),recipes);
                                }
                                $('#time-bar').val(parseInt(cooktime));
                                $('#time').text(cooktime);
                                fitnessButtonSetting(goal);
                                fitnessButtonSetting(state);
                                hideNoParameterMessage();
                                if (bothOrNoneChosen()) {
                                    getSearchResults();
                                } else {
                                    showNeedBothMessage();
                                }
                        } else {
                            showNoParameterMessage();
                        }
                    }
                },
                error: function(xhr, status, error) {
                    console.log('failed');
                },
                complete: function() {
                    $("#home-loading").hide();
                }
            });
        } else {
            console.log('yup');
            $("#no-input-entered").hide();
            $("#home-loading").show();
            setTimeout(check, 150);
        }
    }
    check();

    $(".fit").click(function() {
        hideNeedBothMessage();
        var parent_name = $(this).parent().attr('id');
        if (parent_name === 'weight-buttons') {
            parent_name = 'goal';
        } else {
            parent_name = 'state';
        }
        if ($(this).hasClass("active")) {
            $(this).removeClass("active");
            updateSetting("",parent_name);
        } else {
            $(this).addClass("active");
            $(this).siblings().removeClass("active");
            var name = $(this).attr('id');
            updateSetting(name,parent_name);
        }
        hideNoParameterMessage();
        hideNoResultsMessage();
        if (bothOrNoneChosen()) {
            getSearchResults();
        } else {
            removeSearchResults();
            showNeedBothMessage();
        }
    })
    .mouseup(function() {
        $(this).blur();
    });

    $("#flavor1").tooltip({
        items: '#flavor1',
        content: $(this).title
    }).tooltip("open");
    $("#flavor2").tooltip({
        items: '#flavor2',
        content: $(this).title
    }).tooltip("open");
    
   	$('#recipe-search').submit(function () {
        if (localStorage.getItem("user")!==null) {
        	var recipeName = $('#recipe-name').val();
        	$.ajax({
        		url: "/home/recipeinput",
        		type: "Get",
        		data: {
        			search: recipeName
        		},
        		success: function(recipes) {
                    $("#recipe-list").empty();
                    hideNeedBothMessage();
                    if (recipes === '') {
                        if (!bothOrNoneChosen()) {
                            showNeedBothMessage();
                            removeSearchResults();
                        } else if (allParametersEmpty() && noneChosen()) {
                            showNoParameterMessage();
                            removeSearchResults();
                        } else {
                            hideNoParameterMessage();
                            getSearchResults();
                        }
                    } else {
                        updateSetting(recipes,'recipe');
                        hideNoParameterMessage();
                        hideNoResultsMessage();
                        addSearchParameter($("#recipe-list"),recipes);
                        if (bothOrNoneChosen()) {
                            getSearchResults();
                        } else {
                            showNeedBothMessage();
                        }
                    }
                    $('#recipe-name').val('');
        		},
        		error: function(xhr, status, error) {
        			console.log("Image failed to load");
        		}
        	});
        }
    	return false;
	});

    var ingredients = 
            ["bread",
            "chicken",
            "pasta",
            "tomato",
            "broccoli",
            "spinach",
            "beans",
            "potatoes",
            "pork",
            "eggs",
            "drumstick",
            "kaffir lime",
            "plantain",
            "turnip",
            "sweet potatoes",
            "pimiento",
            "chilli",
            "red chilli",
            "onion",
            "mushroom",
            "radish",
            "garlic",
            "shallots",
            "lettuce",
            "leek",
            "pumpkin",
            "yam",
            "jalapeno",
            "jackfruit",
            "horseradish",
            "spring onion",
            "green peas",
            "green chillies",
            "gherkins",
            "fenugreek",
            "cucumber",
            "zucchini",
            "corn",
            "gai lan",
            "shiitake mushroom",
            "portobello",
            "common mushroom",
            "celery",
            "cauliflower",
            "carrot",
            "capsicum",
            "capers",
            "bottle gourd",
            "bitter gourd",
            "lady finger",
            "lotus stem",
            "bell pepper",
            "beetroot",
            "pigweed",
            "cabbage",
            "bamboo shoot",
            "baby corn",
            "avocado",
            "eggplant",
            "asparagus",
            "ash gourd",
            "artichoke",
            "colocasia",
            "ginger",
            "yeast",
            "nutella",
            "coriander",
            "raspberry",
            "ice cream",
            "chives",
            "galangal",
            "tulsi",
            "sage",
            "rosemary",
            "oregano",
            "nasturtium",
            "salt",
            "mustard powder",
            "paprika",
            "mint leaves",
            "marjoram",
            "lemongrass",
            "provolone cheese",
            "saffron",
            "onion seeds",
            "mace",
            "nutmeg",
            "herbs",
            "thyme",
            "turmeric",
            "garam masala",
            "fenugreek seeds",
            "fennel",
            "green cardamom",
            "dill",
            "curry leaves",
            "cumin seeds",
            "coriander seeds",
            "parsley",
            "acacia",
            "coriander",
            "cloves",
            "cinnamon",
            "star anise",
            "cayenne",
            "caraway seeds",
            "rock salt",
            "black pepper",
            "black cumin",
            "bay leaf",
            "basil",
            "asafoetida",
            "aniseed",
            "allspice",
            "flour",
            "muesli",
            "oats",
            "rice",
            "brown rice",
            "arborio rice",
            "tapioca",
            "semolina",
            "ragi",
            "puffed rice",
            "buckwheat",
            "kidney beans",
            "lentils",
            "couscous",
            "cornmeal",
            "pressed rice",
            "breadcrumbs",
            "chickpeas",
            "basmati rice",
            "barley",
            "sugar",
            "caramel",
            "salt",
            "beef",
            "ground beef",
            "beef stock",
            "minced beef",
            "steak",
            "flank steak",
            "sirloin steak",
            "round steak",
            "tenderloin steak",
            "rib eye steaks",
            "chicken thighs",
            "chicken stock",
            "chicken breasts",
            "chicken liver",
            "chicken drumsticks",
            "whole chicken",
            "turkey",
            "partridge",
            "meat stock",
            "keema",
            "ham",
            "kidney",
            "crab",
            "pork chops",
            "lamb chops",
            "lamb",
            "quail",
            "bacon",
            "gruyere",
            "gouda",
            "feta",
            "milk",
            "brie",
            "cream cheese",
            "ricotta",
            "parmesean",
            "blue cheese",
            "cheddar",
            "mascarpone",
            "cream",
            "bass",
            "mozzarella",
            "khoya",
            "yogurt",
            "cottage cheese",
            "condensed milk",
            "butter",
            "buttermilk",
            "cranberry",
            "kiwi",
            "blueberries",
            "mango",
            "watermelon",
            "strawberry",
            "water chestnut",
            "papaya",
            "orange rind",
            "orange",
            "olives",
            "pear",
            "sultana",
            "mulberry",
            "lychee",
            "lemon juice",
            "lemon rind",
            "lemon",
            "raisins",
            "jamun",
            "tamarind",
            "grapefruit",
            "dried fruit",
            "dates",
            "custard",
            "currant",
            "coconut",
            "cherry",
            "banana",
            "peach",
            "apricots",
            "apple",
            "fig",
            "grape",
            "pomegranate",
            "pineapple",
            "guava",
            "plum",
            "shrimp",
            "tuna",
            "shellfish",
            "shark",
            "hilsa",
            "sardines",
            "salmon",
            "prawns",
            "pomfret",
            "perch",
            "mussel",
            "mullet",
            "squid",
            "haddock",
            "flounder",
            "fish stock",
            "fish",
            "cuttlefish",
            "cod",
            "arugula",
            "clams",
            "catfish",
            "mackerel",
            "anchovies",
            "chia seeds",
            "havelnut",
            "pine nuts",
            "sunflower",
            "sesame oil",
            "pistachio",
            "olive oil",
            "mustard seeds",
            "poppy seeds",
            "sesame seeds",
            "peanuts",
            "cashews",
            "almonds",
            "walnuts",
            "almond milk",
            "red wine",
            "margarine",
            "soy milk",
            "white wine",
            "pepper",
            "rice vinegar",
            "hoisin sauce",
            "malt vinegar",
            "chocolate chips",
            "quinoa",
            "polenta",
            "oyster sauce",
            "vinegar",
            "coconut oil",
            "rice noodles",
            "coffee",
            "beer",
            "chocolate",
            "sake",
            "tortilla",
            "tomato puree",
            "soy sauce",
            "vermicelli",
            "rum",
            "meringue",
            "mayonnaise",
            "jelly",
            "gelatin",
            "cranberry sauce",
            "blackberry",
            "cognac",
            "cocoa",
            "tea",
            "baking soda",
            "tofu",
            "baking powder",
            "lime",
            "eel",
            ];

    $('#ingredient-name').autocomplete({
        minLength: 0,
        appendTo: '#ingredient-autocomplete-container',
        delay: 0,
        source: function(request, response) {
            var results = $.ui.autocomplete.filter(ingredients, request.term);
            response(results.slice(0, 10));
        },
        select: function(event, ui) {
            $("#ingredient-name").val(ui.item.value);
            $("#ingredient-search").submit();
        }
    })
    .blur(function(){
    $(this).autocomplete('enable');
    })
    .focus(function () {
        $(this).autocomplete('search', '');
    });

    $('#ingredient-search').submit(function (){
        if (localStorage.getItem("user")!==null) {
            var ingredientName = $('#ingredient-name').val();
            $.ajax({
                url: "/home/ingredientinput",
                type: "Get",
                data: {
                    search: ingredientName
                },
                success: function(ingredient) {
                    if ($('#ingredient-list li').length < 10) {
                        if (ingredients.indexOf(ingredient) != -1 && !inSearchParameters($('#ingredient-list'),ingredient)) {
                            saveSetting(ingredient,'ingredient');
                            hideNoParameterMessage();
                            hideNoResultsMessage();
                            hideNeedBothMessage();
                            addSearchParameter($('#ingredient-list'),ingredient);
                            if (bothOrNoneChosen()) {
                                getSearchResults();
                            } else {
                                showNeedBothMessage();
                            }
                        }
                    }
                    $('#ingredient-name').val('');
                },
                error: function(xhr, status, error) {
                    console.log("No Ingredient Entered");
                }
            });
        }
        return false;
    });
    $("ul").on("click", "button.remove-input", function () {
        var input = $(this).parent()[0].innerText;
        var list_id = $(this).parent().parent().parent().attr('id');
        if (list_id === 'restriction-list') {
            removeSetting(input, 'restriction');
        } else if (list_id === 'cuisine-list') {
            removeSetting(input, 'cuisine');
        } else if (list_id === 'ingredient-list') {
            removeSetting(input, 'ingredient');
        } else if (list_id === 'recipe-list') {
            updateSetting("", 'recipe');
        }
        $(this).parent().parent().remove();
        hideNoResultsMessage();
        hideNeedBothMessage();
        if (!bothOrNoneChosen()) {
            showNeedBothMessage();
            removeSearchResults();
        } else if (allParametersEmpty() && noneChosen()) {
            showNoParameterMessage();
            removeSearchResults();
        } else {
            getSearchResults();
        }
        return false;
    });
    $($(".ui.icon.save")).tooltip({
        items: '.ui.icon.save',
        content: $(this).title
    }).tooltip("open");

    $(document).on("click", "button.ui.save.small", function () {
        if (localStorage.getItem("user")!==null) {
            var Recipe = $(this).parent().attr('title');
            var but = $(this);
            var user = localStorage.getItem("user");
            var user_id = localStorage.getItem("user_id");
            var thing = [user,Recipe,user_id];
            $.ajax({
                url: "/home/togglesaved",
                type: "Post",
                data: {
                    array: JSON.stringify(thing)
                },
                success: function(data) {
                    if (data === '0') {
                        but.addClass("green");
                        but.attr('title','Remove recipe');
                    } else if (data === '1') {
                        but.removeClass("green");
                        but.attr('title','Save recipe');
                    } else {
                        console.log('Too many saved! Must remove some.');
                    }
                },
                error: function(xhr, status, error) {
                    console.log('Failed to toggle recipe save')
                }
            });
        }
        return false;
    })
    .on("mouseup", "button.ui.save.small", function() {
        $(this).blur();
    });

    $("#ingredient-clear-all").click(function() {
        if (localStorage.getItem("user")!==null) {
            removeAllIngredients();
            $("#ingredient-list").empty();
            hideNoResultsMessage();
            hideNeedBothMessage();
            if (!bothOrNoneChosen()) {
                showNeedBothMessage();
                removeSearchResults();
            } else if (allParametersEmpty() && noneChosen()) {
                showNoParameterMessage();
                removeSearchResults();
            } else {
                getSearchResults();
            }
            return false;
        }
    });

    allergies = 
        ["vegetarian",
        "dairy",
        "egg",
        "gluten",
        "peanut",
        "seafood",
        "sesame",
        "soy",
        "pescetarian",
        "vegan",
        "paleo",
        "sulfite",
        "wheat"];

    $('#restriction-name').autocomplete({
        minLength: 0,
        appendTo: '#restriction-autocomplete-container',
        delay: 0, 
        source: function(request, response) {
            var results = $.ui.autocomplete.filter(allergies, request.term);
            response(results.slice(0, 10));
        },
        select: function(event, ui) {
            $("#restriction-name").val(ui.item.value);
            $("#restrictions-search").submit();
        }
    })
    .blur(function(){
    $(this).autocomplete('enable');
    })
    .focus(function () {
        $(this).autocomplete('search', '');
    });

    $('#restrictions-search').submit(function (){
        if (localStorage.getItem("user")!==null) {
            var restrictionName = $('#restriction-name').val();
            $.ajax({
                url: "/home/restrictioninput",
                type: "Get",
                data: {
                    restrictions: restrictionName
                },
                success: function(restriction) {
                    if ($('#restriction-list li').length < 4) {
                        if (allergies.indexOf(restriction) != -1 && !inSearchParameters($('#restriction-list'),restriction)) {
                            saveSetting(restriction,'restriction');
                            hideNoParameterMessage();
                            hideNoResultsMessage();
                            hideNeedBothMessage();
                            addSearchParameter($('#restriction-list'),restriction);
                            if (bothOrNoneChosen()) {
                                getSearchResults();
                            } else {
                                showNeedBothMessage();
                            }
                        }
                    }
                    $('#restriction-name').val('');
                },
                error: function(xhr, status, error) {
                    console.log("No Restriction Entered");
                }
            });
        }
        return false;
    });

    $('#time-search').change(function() {
        if (localStorage.getItem("user")!==null) {
            var timeBar = $('#time-bar').val();
            updateSetting(timeBar,'cooktime');
            $.ajax({
                url: "/home/timeinput",
                type: "Get",
                data: {
                    times: timeBar
                },
                success: function(time) {
                    hideNoParameterMessage();
                    hideNoResultsMessage();
                    hideNeedBothMessage();
                    if (bothOrNoneChosen()) {
                        getSearchResults();
                    } else {
                        showNeedBothMessage();
                    }
                },
                error: function(xhr, status, error) {
                    console.log("No Time Entered");
                }
            });
        }
        return false;
    });

    var cuisines =
        ["Chinese",
        "American",
        "Italian",
        "Asian",
        "Mexican",
        "French",
        "Barbecue",
        "Southwestern",
        "Indian",
        "English",
        "Mediterranean",
        "Greek",
        "Spanish",
        "German",
        "Thai",
        "Moroccan",
        "Irish",
        "Japanese",
        "Cuban",
        "Hawaiian",
        "Swedish",
        "Hungarian",
        "Portugese"];

    $('#cuisine-name').autocomplete({
        minLength: 0,
        appendTo: '#cuisine-autocomplete-container',
        delay: 0,
        source: function(request, response) {
            var results = $.ui.autocomplete.filter(cuisines, request.term);
            response(results.slice(0, 40));
        },
        select: function(event, ui) {
            $("#cuisine-name").val(ui.item.value);
            $("#cuisine-search").submit();
        }
    })
    .blur(function(){
    $(this).autocomplete('enable');
    })
    .focus(function () {
        $(this).autocomplete('search', '');
    });
    $("#cuisine-search").submit(function (){
        if (localStorage.getItem("user")!==null) {
            var cuisineName = $('#cuisine-name').val();
            $.ajax({
                url: "/home/cuisineinput",
                type: "Get",
                data: {
                    cuisines: cuisineName
                },
                success: function(cuisine) {
                    if ($('#cuisine-list li').length < 5) {
                        if (cuisines.indexOf(cuisine) != -1 && !inSearchParameters($('#cuisine-list'),cuisine)) {
                            saveSetting(cuisine,'cuisine');
                            hideNoParameterMessage();
                            hideNoParameterMessage();
                            hideNeedBothMessage();
                            addSearchParameter($('#cuisine-list'),cuisine);
                            if (bothOrNoneChosen()) {
                                getSearchResults();
                            }   else {
                                showNeedBothMessage();
                            }
                        }
                    }
                    $('#cuisine-name').val('');
                },
                error: function(xhr, status, error) {
                    console.log("No Cuisine Entered");
                }
            });
        }
        return false;
    });
});
