var host = {};
var t_scroll = 1000;
var screen_h = 0;

function set_page() {
	var h = $(window).height() * 1;
	screen_h = h;
}

host.BrowserHost = class {

    constructor() {
        if (!window.ga) {
            window.GoogleAnalyticsObject = 'ga';
            window.ga = window.ga || function() {
                window.ga.q = window.ga.q || [];
                window.ga.q.push(arguments);
            };
            window.ga.l = 1 * new Date();
        }
        window.ga('create', 'UA-54146-13', 'auto');

        window.addEventListener('error', (e) => {
            this.exception(e.error, true);
        });
        window.eval = () => {
            throw new Error('window.eval() not supported.');
        };
    }

    get document() {
        return window.document;
    }

    get version() {
        return this._version;
    }

    get type() {
        return this._type;
    }

    require(id) {
        const url = this._url(id + '.js');
        window.__modules__ = window.__modules__ || {};
        if (window.__modules__[url]) {
            return Promise.resolve(window.__exports__[url]);
        }
        return new Promise((resolve, reject) => {
            window.module = { exports: {} };
            let script = document.createElement('script');
            script.setAttribute('id', id);
            script.setAttribute('type', 'text/javascript');
            script.setAttribute('src', url);
            script.onload = () => {
                let exports = window.module.exports;
                delete window.module;
                window.__modules__[id] = exports;
                resolve(exports);
            };
            script.onerror = (e) => {
                delete window.module;
                reject(new Error('The script \'' + e.target.src + '\' failed to load.'));
            };
            this.document.head.appendChild(script);
        });
    }
}

window._host_ = new host.BrowserHost();
$(function(){
    $(window).resize(set_page);
	$(window).load(set_page);

	var scrollTop = $(window).scrollTop();
	var is_scroll = false;
    
	$(window).scroll(function () {	
        scrollTop = $(window).scrollTop();
        if (scrollTop > 0 && !is_scroll) {
            $("#text-container").animate({ marginTop: '-100px' }, t_scroll);
            is_scroll = true;
        } else if (scrollTop == 0) {
            $("#text-container").animate({ marginTop: '0px' }, t_scroll);
            is_scroll = false;
        }
	});
	
	$('body').on("click", '#myBtn', function(){
        $('html, body').animate({scrollTop: screen_h-80}, t_scroll);
        $("#text-container").animate({ marginTop: '-100px' }, t_scroll);
		is_scroll = true;
		return false;
    });
});

// $('#img1'),addEventListener("mouseover", mouseOver);
// $('#img1'),addEventListener("mouseout", mouseOut);
function mouseOver(num) {
    var index_img = "#img";
    index_img = index_img + num.toString(10);
    $(index_img).animate({ top: '-15px' }, 10);
    $(index_img+'_tip').css('display', 'block');
    $(index_img+'_top').css('display', 'block');
}
function mouseOut(num) {
    var index_img = "#img";
    index_img = index_img + num.toString(10);
    $(index_img).animate({ top: '0px' }, 10);
    $(index_img+'_tip').css('display', 'none');
    $(index_img+'_top').css('display', 'none');
}
