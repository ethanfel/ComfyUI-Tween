import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

app.registerExtension({
    name: "Tween.VideoPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "TweenConcatVideos") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const container = document.createElement("div");
            const previewWidget = this.addDOMWidget("videopreview", "preview", container, {
                serialize: false,
                hideOnZoom: false,
                getValue() { return container.value; },
                setValue(v) { container.value = v; },
            });

            previewWidget.computeSize = function (width) {
                if (this.aspectRatio && !this.videoEl.hidden) {
                    const height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
                    return [width, height > 0 ? height : -4];
                }
                return [width, -4];
            };

            const previewNode = this;

            previewWidget.videoEl = document.createElement("video");
            previewWidget.videoEl.controls = true;
            previewWidget.videoEl.loop = true;
            previewWidget.videoEl.muted = true;
            previewWidget.videoEl.style.width = "100%";
            previewWidget.videoEl.hidden = true;

            previewWidget.videoEl.addEventListener("loadedmetadata", () => {
                previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
                fitHeight(previewNode);
            });
            previewWidget.videoEl.addEventListener("error", () => {
                previewWidget.videoEl.hidden = true;
                fitHeight(previewNode);
            });

            container.appendChild(previewWidget.videoEl);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            if (!message?.gifs?.length) return;

            const params = message.gifs[0];
            const previewWidget = this.widgets?.find((w) => w.name === "videopreview");
            if (!previewWidget) return;

            const query = new URLSearchParams(params);
            query.set("timestamp", Date.now());
            previewWidget.videoEl.src = api.apiURL("/view?" + query);
            previewWidget.videoEl.hidden = false;
            previewWidget.videoEl.autoplay = true;
        };
    },
});
