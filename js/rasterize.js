var page = require('webpage').create(), address, output, size;

if (phantom.args.length < 2 || phantom.args.length > 3) {
	console.log('Usage: rasterize.js URL filename');
	phantom.exit();
} else {
	address = phantom.args[0];
	output = phantom.args[1];
	page.viewportSize = { width: 1024, height: 768 };
	page.clipRect = { top: 0, left: 0, width: 1024, height: 768 }
	page.onError = function (msg, trace) {
		console.log(msg);
		trace.forEach(function(item) {
			console.log('  ', item.file, ':', item.line);
		});
		phantom.exit(2);
	}
	page.open(address, function (status) {
		if (status !== 'success') {
			console.log('Unable to load the address!');
			phantom.exit(1);
		} else {
			window.setTimeout(function () {
				page.render(output);
				phantom.exit();
			}, 5000);
		}
	});
}