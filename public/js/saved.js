$(document).ready(function() {
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


    var check = function () {
        if (localStorage.getItem("user")!==null) {
        	var user = localStorage.getItem("user");
            var user_id = localStorage.getItem("user_id");
            var thing = [user, user_id];
            $.ajax({
                url: "/saved/display",
                type: "Get",
                data: {
                    array: JSON.stringify(thing)
                },
                beforeSend: function() {
                    $("#loading-message").show();
                },
                success: function(data) {
                    var saved = data;
                    if (saved.length === 0) {
                        $("#no-recipes-message").show();
                    } else {
                        for (i=0; i<saved.length;i++) {
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
                                title: saved[i]
                            });
                            var level4span2 = $("<span/>", {
                                class: "floated like flavors"
                            });
                            var level5button = $("<button/>", {
                                class:"green ui icon button save small",
                                title: "Remove recipe"
                            });
                            var level6i = $("<i/>", {
                                class:"remove icon"
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
                            getRecipeDetails(saved[i],level1div);
                        }
                    }
                },
                error: function(xhr, status, error) {
                    console.log('Recipes failed to load!');
                },
                complete: function() {
                    $("#loading-message").hide();
                }
            });
        } else {
            console.log('yup');
            $("#loading-message").show();
            setTimeout(check, 150);
        }
    }
    check();

	$(document).on("click", "button.ui.small", function() {
		var Recipe = $(this).parent().attr('title');
		var user = localStorage.getItem("user");
        var user_id = localStorage.getItem("user_id");
		$(this).parent().parent().parent().parent().remove();
		$.ajax({
			url: "/saved/remove",
			type: "Post",
			data: {
				array: JSON.stringify([user,Recipe,user_id])
			},
			success: function(data) {
				console.log('removed');
			},
			error: function(xhr, status, error) {
				console.log('failed to remove');
			}
		});
		return false;
	});
    $("#flavor1").tooltip({
        items: $(this),
        content: $(this).title
    }).tooltip("open");	
    $("#flavor2").tooltip({
        items: $(this),
        content: $(this).title
    }).tooltip("open"); 
});