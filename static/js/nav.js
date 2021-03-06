﻿// TODO
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
        // area: {en:"Area", ru: "Область"},
        // log: {en:"Log", ru: "Log"},
        about: {en:"About", ru: "О системе"},

        }  

const divToastersPlace = `
                <div aria-live="polite" aria-atomic="true" style="position: relative; min-height: 20px;  z-index: 10">
                    <div style="position: absolute; top: 10; right: 20;" id="toastersPlace">
                    </div>
                </div>
            `;

const divToaster = `
            <div class="toast" role="alert" hide.bs.toast aria-live="assertive" aria-atomic="true" data-delay="5000">
                <div class="toast-header">
                    <strong class="mr-auto">Information</strong>
                    <button type="button" class="ml-2 mb-1 close" data-dismiss="toast" aria-label="Close">
                        <span aria-hidden="true">&times;</span>
                    </button>
                </div>
                <div class="toast-body"></div>
            </div>
            `
function showToaster(msg) {
    console.log("showToaster");
    var t = $(divToaster);
    t.on('hide.bs.toast', function () { this.remove(); });
    t.find('.toast-body').append(msg);
    $('#toastersPlace').append(t);
    t.toast('show');
}

const divNav = `
       <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <a class="navbar-brand" href="#">Wingu:Main</a>
            <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarColor01" aria-controls="navbarColor01" aria-expanded="false" aria-label="Toggle navigation">
                <span class="navbar-toggler-icon"></span>
            </button>

            <div class="collapse navbar-collapse" id="navbarColor01">
                <ul class="navbar-nav mr-auto">
                    <li class="nav-item" id='menu_index'><a class="nav-link text" onclick="doMenu(this.id)" href="#" id='index'><span class="sr-only">(current)</span></a></li>
                    <li class="nav-item" id='menu_cameras'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='cameras'></a></li>
                    
                    <li class="nav-item" id='menu_server'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='server'></a></li>
                    <li class="nav-item" id='menu_config'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='config'></a></li>

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
// <li class="nav-item" id='menu_log'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='log'></a></li>
// <li class="nav-item" id='menu_file'><a class="nav-link text" href="#" onclick="doMenu(this.id)" id='file'></a></li>

function translate() {
    console.log("translate", curLang, curId )
    if (langText.hasOwnProperty(curId)) {
        $(".text").each(function (index) {
            $(this).text(langText[$(this)[0].id][curLang]);
        });
        // console.log("translate",langText[curId][curLang]);
        $('#headMain').text(langText[curId][curLang])
    } else {
        console.log("translate", curLang, curId);
    }
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
    if (localStorage.getItem('myId') != null) { curId = localStorage.getItem('myId'); }
    let path = window.location.pathname.split('/');
    curId = path[path.length - 1];
    curId = curId.substring(0, curId.length - 5)
    console.log("Start main page " + curId);
    const navigation = document.createElement('div');
    navigation.className = 'menuContent';
    navigation.innerHTML = divNav;           
    $('#menu').append(navigation);
    $("#langSwitch").html('<span class="flag-icon flag-icon-'+curLang+'"></span>'+curLang);
    $('#menu_' + curId).addClass("active");
    console.log("Start page ", curId, curLang);
    translate();

    // $("#menu_Home").text("ffff");
    $("#dropdown-menu-lang").on('click', 'a', function(){
            curLang =  $(this)[0].id;
            // console.log($("#dropdown").text());
            $("#langSwitch").html('<span class="flag-icon flag-icon-'+curLang+'"></span>'+curLang);
            localStorage.setItem('myLang', curLang);
            translate();
    });

    $('#menu').append(divToastersPlace);
    showToaster("<span style='color:blue'><b>Script start!</b></span>");

    });