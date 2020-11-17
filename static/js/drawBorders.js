//<line class="static" id="0_line" x1="200" y1="200" x2="100" y2="100" stroke-width="2" stroke="#00ff21" / >
//<line class="static" id="0_line_back" x1="200" y1="204" x2="100" y2="104" stroke-width="2" stroke="#ff0021" / >
//<circle class="draggable" id="0_line_start" cx="200" cy="200" r="5" stroke="blue" stroke-width="1" fill="#00ff21" / >
//<circle class="draggable" id="0_line_end" cx="100" cy="100" r="5" stroke="blue" stroke-width="1" fill="#ff0021" / >

var drawBorders = {};
var drawActiveBorder = null;
var drawBorderCircleradius = 6;

var css = '.static {} .draggable {cursor: move;}',
    head = document.head || document.getElementsByTagName('head')[0],
    style = document.createElement('style');

head.appendChild(style);

style.type = 'text/css';
if (style.styleSheet) {style.styleSheet.cssText = css;
} else { style.appendChild(document.createTextNode(css));
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
    let svg = $("#fileSvg")[0];
    let len = Math.floor(Math.sqrt(Math.pow(xy[2] - xy[0], 2) + Math.pow(xy[1] - xy[3], 2)));
    var angleDeg = Math.floor(Math.atan2(xy[3] - xy[1], xy[2] - xy[0]) * 180 / Math.PI) - 90;
    drawNewRectSvg(svg, id.toString() + '_line_back', xy[0], xy[1], 6, len, angleDeg.toString(), "#ff0021");
    drawNewLineSvg(svg, id.toString() + '_line', xy[0], xy[1], xy[2], xy[3], "#00ff21");
    //drawNewLineSvg(svg, id.toString() + '_line_back', x1 - dist[1], y1 + dist[0], x2 - dist[1], y2 + dist[0], "#ff0021");
    drawNewCircleSvg(svg, id.toString() + "_line_start", xy[0], xy[1], "#00ff21");
    drawNewCircleSvg(svg, id.toString() + "_line_end", xy[2], xy[3], "#ff0021");
}


function addNewBorder(name, data) {
    console.log(name);
    if (!name) {
        name = guidGenerator(1, drawBorders, 0);
        data = [200, 200, 160, 100];
    }
    if (name != "") {
        drawBorder(name, data);
        drawBorders[name] = data;
        console.log(drawBorders);
        $('#dtVerticalScroll tbody').append('<tr style="cursor:pointer;" id="' + name + '_line_sel" onclick="setActiveBorder(this.id)"><td>' + name + '</td></tr>');
        setActiveBorder(name + "_line");
    } else {
        console.log("Error create uniq name!");
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


function makeDraggable(evt) {
    var LongLine, LongLine2, startOrNot;
    var svg = evt.target;
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
        var CTM = svg.getScreenCTM();
        if (evt.touches) { evt = evt.touches[0]; }
        return {
            x: (evt.clientX - CTM.e) / CTM.a,
            y: (evt.clientY - CTM.f) / CTM.d
        };
    }

    function startDrag(evt) {
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
            drawBorders[id] = [x1, y1, x2, y2];
            console.log(drawBorders);
            selectedElement = false;
            LongLine = null;
            LongLine2 = null;
        }
    }
}