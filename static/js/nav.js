// TODO
//

var curLang = 'en';
var curId = 'index';
var langPack = {

        };
        
var langText = {
        index: {en:"Home", ru: "Главная"},
        cameras: {en:"Cameras", ru: "Камеры"},
        config: {en:"Config", ru: "Настройки"},
        server: {en:"Server", ru: "Сервер"},
        file: {en:"File", ru: "Файл"},
        log: {en:"Log", ru: "Log"},
        about: {en:"About", ru: "О системе"},

        }  

const divNav = `
       <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <a class="navbar-brand" href="#">WihguMain</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarColor01" aria-controls="navbarColor01" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>

            <div class="collapse navbar-collapse" id="navbarColor01">
                <ul class="navbar-nav mr-auto">
                    <li class="nav-item" id='menu_index'><a class="nav-link text" onclick="doMenu(this.id)" href="#" id='index'><span class="sr-only">(current)</span></a></li>
                    <li class="nav-item" id='menu_cameras'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='cameras'></a></li>
                    <li class="nav-item" id='menu_file'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='file'></a></li>
                    <li class="nav-item" id='menu_server'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='server'></a></li>
                    <li class="nav-item" id='menu_config'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='config'></a></li>
                    <li class="nav-item" id='menu_log'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='log'></a></li>
                    <li class="nav-item" id='menu_about'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='about'></a></li>
                </ul>
                <div class="form-inline">
                    <div class="nav-item dropdown">
                        <a class="nav-link dropdown-toggle" href="#" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false" id="langSwitch"><span class="flag-icon flag-icon-en"></span>en</a>
                        <div class="dropdown-menu" id="dropdown-menu-lang" aria-labelledby="dropdown09">
                            <a class="dropdown-item" href="#en" id="en"><span class="flag-icon flag-icon-en"></span>en</a>
                            <a class="dropdown-item" href="#ru" id="ru"><span class="flag-icon flag-icon-ru"></span>ru</a>
                        </div>
                    </div>

                    <a class="nav-link" href="#" onclick="doSignOut()" id="loginType">Sign out</a>
                </div>
                <!-- <form class="form-inline">
                    <input class="form-control mr-sm-2" type="search" placeholder="Search" aria-label="Search">
                    <button class="btn btn-outline-info my-2 my-sm-0" type="submit">Search</button>
                </form> -->
            </div>
        </nav>
  `;
 
  function translate(){
            $(".text").each(function(index) {
                $(this).text(langText[$(this)[0].id][curLang]);
            });
            $('#headMain').text(langText[curId][curLang])
        }
        
  function doMenu(id){
            // console.log("Click event is triggered on the link.", id, window.location.href);
            window.location.href = id+".html";
            localStorage.setItem('myId', id);
        }
        
  function doSignOut(){
            $.post('/sign/1', {'login': ''}, function(data){
                console.log(data);
                if (data.error){showError(data.error)
                }else{
                console.log("redirect to root.")
                window.location.href = '/'}
            });
        }

$(document).ready(function() {
        if(localStorage.getItem('myLang') != null ){ curLang = localStorage.getItem('myLang');}
        if(localStorage.getItem('myId') != null ){ curId = localStorage.getItem('myId');}
        console.log("Start main page "+curId);
        const navigation = document.createElement('div');
        navigation.className = 'menuContent';
        navigation.innerHTML = divNav;           
        $('#menu').append(navigation);
        $("#langSwitch").html('<span class="flag-icon flag-icon-'+curLang+'"></span>'+curLang);
        $('#menu_'+curId).addClass("active");
        translate();
        // $("#menu_Home").text("ffff");
        $("#dropdown-menu-lang").on('click', 'a', function(){
                curLang =  $(this)[0].id;
                // console.log($("#dropdown").text());
                $("#langSwitch").html('<span class="flag-icon flag-icon-'+curLang+'"></span>'+curLang);
                localStorage.setItem('myLang', curLang);
                translate();
            });
        
        });