// TODO
// add move each new border a liitle right from prev
// add possibility save and load borders templates

// var fileCurVideoFilePath = 'video/39.avi';
// json data structure:  name, size in GB, creaction time
// var fileList = {};
// var fileBordersList = {};

var fileImgs = {};
var tmp_config_borders = {}
var drawBorders = {};
var drawActiveBorder = null;
var drawBorderCircleradius = 6;

var css = '.static {} .draggable {cursor: move;}',
    head = document.head || document.getElementsByTagName('head')[0],
    style = document.createElement('style');

head.appendChild(style);

style.type = 'text/css';
if (style.styleSheet) {
    style.styleSheet.cssText = css;
} else {
    style.appendChild(document.createTextNode(css));
}


function validedName() {
    let input = $('#new_border_name');
    let c = input.selectionStart, r = /[^a-zA-Z0-9_]/gi, v = input.val();
    if (r.test(v)) {
        input.val(v.replace(r, ''));
        c--;
    }
}

function showImgBorders() {
    console.log("showImgBorders", curCamId);
    let streamsConfigList = JSON.parse(localStorage.getItem('streamsConfigList'));
    let isNotExist = false;
    if (curCamId == undefined) {
        let cam_id = $('#addStream_id').val();
        if (streamsConfigList.hasOwnProperty(cam_id)) {
            alert("This stream id already exist! Please will give other name.");
        } else {
            isNotExist = true;
            tmp_config_borders = {};
        }         
    }
    if (isNotExist) {
        //drawBorders = 
        if ($("#result_image").src != "" && $('#addStream_url').val() != "") {
            let fileCurVideoFilePath = $('#addStream_url').val();
            // console.log(!(fileCurVideoFilePath in fileImgs), fileImgs);
            if (!(fileCurVideoFilePath in fileImgs)) {
                getFileImgAjax(fileCurVideoFilePath);
            }
            
            if (curCamId != undefined) {
                console.log("streamsConfigList[curCamId]['borders']", streamsConfigList[curCamId]['borders'])
                drawBorders = streamsConfigList[curCamId]['borders'];
                for (var key in drawBorders) {
                    console.log("showImgBorders", key, drawBorders[key])
                    let data = [
                        [parseInt(drawBorders[key][0][0] * 1.3), drawBorders[key][0][1]],
                        [parseInt(drawBorders[key][1][0] * 1.3), drawBorders[key][1][1]]
                    ];
                    console.log("data", data);
                    drawNewBorder(key, data);
                }
                console.log(drawBorders);
            } else {
                console.log("new border"); 
            }
            $('#fileModalImg').modal('show');
            makeDraggable();
        } else {
            console.log("Can not load Image!");
            //alert("Can not load Image!");
        }
    }
}

function getFileImgAjax() {
    console.log("getFileImgAjax", curCamId, $('#addStream_url').val());
    if ($('#addStream_url').val() !="") {
        let filePath = $('#addStream_url').val();
        var jqxhr = $.ajax({
            url: "http://localhost:8080/getFileImg?file=" + filePath,
            cache: false,
            xhrFields: { responseType: 'blob' }
        })
        // xhr.overrideMimeType("text/plain; charset=x-user-defined");
        .done(function (data) {
            // console.log("success ajax", data);
            var url = window.URL || window.webkitURL;
            fileImgs[filePath] = url.createObjectURL(data);
            $('#result_image').attr("src", fileImgs[filePath]);
            $('#fileModalImgDiv').css("background-image", "url('" + fileImgs[filePath] + "')");
        })
        .fail(function () { console.log("error get ajax"); })
        .always(function () { console.log("complete ajax request"); });
    } else {
        console.log("Can not load file");
        alert("Error!\nWrong video path.");
    }
    }

function fileSaveBorders() {
    let streamsConfigList = JSON.parse(localStorage.getItem('streamsConfigList'));
    if (streamsConfigList.hasOwnProperty(curCamId)) {
        drawBorders_old = streamsConfigList[curCamId]['borders'];
        //    console.log(drawBorders_old, drawBorders);
        //if (drawBorders_old != drawBorders) {
        } else {   }
        for (var key in drawBorders) {
            //console.log("showImgBorders", key, drawBorders[key])
            drawBorders[key][0][0] = parseInt(drawBorders[key][0][0] / 1.3);
            drawBorders[key][1][0] = parseInt(drawBorders[key][1][0] / 1.3);
        };
    console.log("cam_id", curCamId)
    if (curCamId != undefined) {
        streamsConfigList[curCamId]['borders'] = drawBorders;
        localStorage.setItem('streamsConfigList', JSON.stringify(streamsConfigList));
        $('#addStream_borders_number').val(Object.keys(streamsConfigList[curCamId]['borders']).length);
    } else {
        tmp_config_borders = drawBorders;
        $('#addStream_borders_number').val(Object.keys(tmp_config_borders).length);
    }
}

function fileCancelBorders() {
    console.log('fileCancelBorders');
    drawBorders = {};
    drawActiveBorder = null;
    $('#fileSvg').empty();
    $('#fileSvg').empty();
    $('#displayLineName').html('&nbsp;');
    $('#dtVerticalScroll tbody').empty();

}

function saveToServer() {
    if (parseInt($("#addStream_borders_number").val()) > 0) {
        console.log("add to queue");
        let streamsConfigList = JSON.parse(localStorage.getItem('streamsConfigList'));
        let defaultConfig = {
            'url': 'video/39.avi',
            'isFromFile': false,
            'save_path': 'video/39_out.avi',
            'body_min_w': 64,
            'path_track': 20,
            'body_res': [256, 128],
            'display_video_flag': true,
            'max_cosine_distance': 0.2,
            'save_video_flag': false,
            'type': 0, // 0 - stream, 1 - area 
            'skip_frames': 0,
            'encoder': 'gdet',
            'encoder_filename': 'mars-small128.pb',
            'batch_size': 32,
            'img_size_start': [1600, 1200],
            'save_video_res': [720, 540],
            'borders': {}
        };
        let isNew = false;
        //id, url, skip_frames, isFromFile, save_path, save_video_res, save_video_flag
        if (curCamId in streamsConfigList) {
            if (streamsConfigList[curCamId].hasOwnProperty('url')) {
                console.log("curCamId");
                defaultConfig = streamsConfigList[curCamId];
            }
        } else {
            isNew = true;
            defaultConfig['borders'] = tmp_config_borders;
        }
        //for (var key in defaultConfig) {
        // console.log("$('#addStream_isFromFile').val()", $('#addStream_isFromFile').prop('checked'))
        // defaultConfig['id'] = $('#addStream_id').val();
        // addStream_save_video_flag
        let name = $('#addStream_id').val();
        let autostart = $('#addStream_autostart').prop('checked');
        defaultConfig['name'] = $('#addStream_name').val();
        defaultConfig['url'] = $('#addStream_url').val();
        defaultConfig['type'] = $('#addStream_type').val();
        defaultConfig['skip_frames'] = parseInt($('#addStream_skip_frames').val());
        defaultConfig['isFromFile'] = $('#addStream_isFromFile').prop('checked');
        defaultConfig['save_path'] = $('#addStream_save_path').val();
        defaultConfig['save_video_flag'] = $('#addStream_save_video_flag').prop('checked');
        //}
        console.log("defaultConfig ready", defaultConfig);
        // streamsConfigList[curCamId] = defaultConfig;
        // localStorage.setItem('streamsConfigList', JSON.stringify(streamsConfigList));
        if (wsCmd == null) { WebSocketCmd();}
         //wsCmd.send(JSON.stringify({ "cmd": "getManagerData" }));

        wsCmd.send(JSON.stringify({ "cmd": "saveStream", "config": { "name": name, "tp": 'Stream_', "data": defaultConfig, "autostart": autostart, "isNew": isNew} }));
    } else {
        alert("Please add some borders!!!!");
    }
}

function guidGenerator(len, list, cnt) {
    let res = '';
    for (i = 0; i < len; i++) { res += (((1 + Math.random()) * 0x10000) | 0).toString(16).substring(1); };
    if (res in list) { if (cnt < 6) { res = guidGenerator(len, list, cnt); } else { return ""; } }
    return res;
}

function drawNewLineSvg(svg, id, x1, y1, x2, y2, color) {
    var obj = document.createElementNS("http://www.w3.org/2000/svg", "line");
    obj.setAttributeNS(null, "x1", x1);
    obj.setAttributeNS(null, "y1", y1);
    obj.setAttributeNS(null, "x2", x2);
    obj.setAttributeNS(null, "y2", y2);
    obj.setAttributeNS(null, "class", "static");
    obj.setAttributeNS(null, "id", id);
    obj.setAttributeNS(null, "stroke", color);
    obj.setAttributeNS(null, "stroke-width", 4);
    obj.onclick = function () {
        setActiveBorder(id);
    };
    svg.appendChild(obj);
}

function drawNewRectSvg(svg, id, x, y, width, height, rot, color) {
    var obj = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    obj.setAttributeNS(null, "x", x);
    obj.setAttributeNS(null, "y", y);
    obj.setAttributeNS(null, "width", width);
    obj.setAttributeNS(null, "class", "static");
    obj.setAttributeNS(null, "id", id);
    obj.setAttributeNS(null, "stroke", color);
    obj.setAttributeNS(null, "fill", 'red');
    obj.setAttributeNS(null, "height", height);
    obj.setAttributeNS(null, "transform", 'rotate(' + rot.toString() + ' ' + x.toString() + ' ' + y.toString() + ')');
    obj.onclick = function () { setActiveBorder(id); };
    svg.appendChild(obj);
}

function drawNewCircleSvg(svg, id, cx, cy, color) {
    var obj = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    obj.setAttributeNS(null, "cx", cx);
    obj.setAttributeNS(null, "cy", cy);
    obj.setAttributeNS(null, "r", drawBorderCircleradius);
    obj.setAttributeNS(null, "class", "draggable");
    obj.setAttributeNS(null, "id", id);
    obj.setAttributeNS(null, "stroke", "blue");
    obj.setAttributeNS(null, "stroke-width", 1);
    obj.setAttributeNS(null, "fill", color);
    svg.appendChild(obj);
}

function setActiveBorder(id) {
    if (id.includes("_back")) { id = id.substring(0, id.indexOf("_back")); }
    if (id.includes("_sel")) { id = id.substring(0, id.indexOf("_sel")); }

    if (drawActiveBorder != null) {
        document.getElementById(drawActiveBorder + "_line_start").setAttribute("r", drawBorderCircleradius);
        document.getElementById(drawActiveBorder + "_line_end").setAttribute("r", drawBorderCircleradius);
    }

    drawActiveBorder = id.substring(0, id.indexOf("_line"));
    document.getElementById('displayLineName').innerHTML = drawActiveBorder;
    document.getElementById(id + "_start").setAttribute("r", drawBorderCircleradius * 2);
    document.getElementById(id + "_end").setAttribute("r", drawBorderCircleradius * 2);
}

function drawBorder(id, xy) {
    //xy = x1, y1, x2, y2
    // console.log("xy",xy);
    let svg = $("#fileSvg")[0];
    let len = Math.floor(Math.sqrt(Math.pow(xy[1][0] - xy[0][0], 2) + Math.pow(xy[0][1] - xy[1][1], 2)));
    //let len = Math.floor(Math.sqrt(Math.pow(xy[2] - xy[0], 2) + Math.pow(xy[1] - xy[3], 2)));
    var angleDeg = Math.floor(Math.atan2(xy[1][1] - xy[0][1], xy[1][0] - xy[0][0]) * 180 / Math.PI) - 90;
    drawNewRectSvg(svg, id.toString() + '_line_back', xy[0][0], xy[0][1], 6, len, angleDeg.toString(), "#ff0021");
    drawNewLineSvg(svg, id.toString() + '_line', xy[0][0], xy[0][1], xy[1][0], xy[1][1], "#00ff21");
    //drawNewLineSvg(svg, id.toString() + '_line_back', x1 - dist[1], y1 + dist[0], x2 - dist[1], y2 + dist[0], "#ff0021");
    drawNewCircleSvg(svg, id.toString() + "_line_start", xy[0][0], xy[0][1], "#00ff21");
    drawNewCircleSvg(svg, id.toString() + "_line_end", xy[1][0], xy[1][1], "#ff0021");
}

function drawNewBorder(name, data) {
    drawBorder(name, data);
    drawBorders[name] = data;
    console.log(drawBorders);
    $('#dtVerticalScroll tbody').append('<tr style="cursor:pointer;" id="' + name + '_line_sel" onclick="setActiveBorder(this.id)"><td>' + name + '</td></tr>');
    setActiveBorder(name + "_line");
}

function addNewBorder(name) {
    if (name) {
        console.log("name", name, drawBorders);
        if (name in drawBorders) {
            alert("Name already exist!!");
        } else {
            console.log("name=", name);
            data = [[200, 200], [160, 100]];
            drawNewBorder(name, data);
            $('#borderModalName').modal('hide');
        }
    } else {
        alert("Please give a border name!!");
    }
    
}


function delActiveBorder() {
    if (drawActiveBorder) {
        //console.log(drawActiveBorder);
        $('#' + drawActiveBorder + '_line').remove();
        $('#' + drawActiveBorder + '_line_back').remove();
        $('#' + drawActiveBorder + '_line_start').remove();
        $('#' + drawActiveBorder + '_line_end').remove();
        $('#displayLineName').html('&nbsp;');
        delete drawBorders[drawActiveBorder];
        //console.log($('#' + drawActiveBorder + '_line_sel'));
        $('#' + drawActiveBorder + '_line_sel').remove();
        drawActiveBorder = null;
        //console.log(drawBorders);
    }
}


function makeDraggable() {
    console.log("makeDraggable");
    let LongLine, LongLine2, startOrNot;
    // var svg = document.getElementById("fileSvg");
    let svg = $("#fileSvg")[0];
    svg.addEventListener('mousedown', startDrag);
    svg.addEventListener('mousemove', drag);
    svg.addEventListener('mouseup', endDrag);
    svg.addEventListener('mouseleave', endDrag);
    svg.addEventListener('touchstart', startDrag);
    svg.addEventListener('touchmove', drag);
    svg.addEventListener('touchend', endDrag);
    svg.addEventListener('touchleave', endDrag);
    svg.addEventListener('touchcancel', endDrag);
    var selectedElement, offset, transform;

    function getMousePosition(evt) {
        // console.log("getMousePosition");
        var CTM = svg.getScreenCTM();
        if (evt.touches) { evt = evt.touches[0]; }
        return {
            x: (evt.clientX - CTM.e) / CTM.a,
            y: (evt.clientY - CTM.f) / CTM.d
        };
    }

    function startDrag(evt) {
        // console.log("Start Drag")
        if (evt.target.classList.contains('draggable')) {
            selectedElement = evt.target;
            //console.log("evt.target", evt.target.id);
            let n = evt.target.id.lastIndexOf("_");
            let lineName = evt.target.id.substring(0, n);
            setActiveBorder(lineName);
            LongLine = document.getElementById(lineName);
            LongLine2 = document.getElementById(lineName + '_back');
            selectedElement.id.includes("start") ? startOrNot = true : startOrNot = false;
            offset = getMousePosition(evt);
            transforms = selectedElement.transform.baseVal;
            if (transforms.length === 0 || transforms.getItem(0).type !== SVGTransform.SVG_TRANSFORM_TRANSLATE) {
                var translate = svg.createSVGTransform();
                translate.setTranslate(0, 0);
                selectedElement.transform.baseVal.insertItemBefore(translate, 0);
            }
            transform = transforms.getItem(0);
            offset.x -= transform.matrix.e;
            offset.y -= transform.matrix.f;
        }
    }

    function drag(evt) {
        // console.log("drag function")
        if (selectedElement) {
            evt.preventDefault();
            var coord = getMousePosition(evt);
            var dx = coord.x - offset.x;
            var dy = coord.y - offset.y;
            transform.setTranslate(dx, dy);
            let x1, y1, x2, y2;
            if (startOrNot) {
                x1 = coord.x;
                y1 = coord.y;
                x2 = parseInt(LongLine.getAttribute('x2'));
                y2 = parseInt(LongLine.getAttribute('y2'));
                LongLine.setAttribute('x1', x1);
                LongLine.setAttribute('y1', y1);
            } else {
                x1 = parseInt(LongLine.getAttribute('x1'));
                y1 = parseInt(LongLine.getAttribute('y1'));
                x2 = coord.x;
                y2 = coord.y;
                LongLine.setAttribute('x2', x2);
                LongLine.setAttribute('y2', y2);
            }
            let len = Math.floor(Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y1 - y2, 2)));
            var rot = Math.floor(Math.atan2(y2 - y1, x2 - x1) * 180 / Math.PI) - 90;
            LongLine2.setAttribute('x', x1);
            LongLine2.setAttribute('y', y1);
            LongLine2.setAttribute("height", len);
            LongLine2.setAttribute("transform", 'rotate(' + rot.toString() + ' ' + x1.toString() + ' ' + y1.toString() + ')');
        }
    }

    function endDrag(evt) {
        if (LongLine) {
            //console.log(evt, LongLine);
            let x1 = parseInt(LongLine.getAttribute('x1'));
            let y1 = parseInt(LongLine.getAttribute('y1'));
            let x2 = parseInt(LongLine.getAttribute('x2'));
            let y2 = parseInt(LongLine.getAttribute('y2'));
            let id = LongLine.id.substring(0, LongLine.id.indexOf('_line'));
            drawBorders[id] = [[x1, y1], [x2, y2]];
            console.log(drawBorders);
            selectedElement = false;
            LongLine = null;
            LongLine2 = null;
        }
    }
}