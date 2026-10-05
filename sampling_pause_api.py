"""ComfyUI API integration; polling keeps frontend independent of preview events."""
from aiohttp import web

def register_routes(prompt_server, controller):
    @prompt_server.routes.post('/sampling_pause')
    async def pause(request):
        accepted = controller.request_pause()
        return web.json_response(controller.status(), status=200 if accepted else 409)

    @prompt_server.routes.post('/sampling_resume')
    async def resume(request):
        controller.release('resume')
        return web.json_response(controller.status())

    @prompt_server.routes.get('/sampling_pause/status')
    async def status(request):
        return web.json_response(controller.status())

    @web.middleware
    async def clear_queue(request, handler):
        # Preserve the original body for the standard Comfy queue handler.
        if request.method == 'POST' and request.path.rstrip('/') in ('/queue', '/api/queue'):
            try:
                body = await request.json()
            except (ValueError, TypeError):
                body = None
            if isinstance(body, dict) and body.get('clear') is True:
                controller.release('queue cleared')
        return await handler(request)

    async def shutdown(app):
        controller.release('shutdown')

    prompt_server.app.middlewares.append(clear_queue)
    prompt_server.app.on_shutdown.append(shutdown)
