

document.addEventListener('DOMContentLoaded', function() {
	
let div = document.createElement('div');
document.body.appendChild( div );
div.id = 'helperbar';
div.style.display = 'block';
div.style.position = 'fixed';
div.style.bottom = '20px';
div.style.left = '20px';
div.style['margin-left'] = '0px';
div.style['z-index'] = '200000';

div.innerHTML = `<a href="https://f21dl.github.io"><img src="https://f21dl.github.io/material/images/webhome.png" height="40px"></a>`;
/*
<div style="display: block; position: fixed; bottom: 20px; left: 20px; margin-left: 0px; z-index: 200000;">
	<a href="https://f21dl.github.io"><img src="https://f21dl.github.io/material/images/webhome.png" height="40px"></a>
</div>
*/

}, false);