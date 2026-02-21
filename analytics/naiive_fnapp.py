
app = func.FunctionApp()


# @app.route(route="/", auth_level="anonymous")
# @app.function_name(name="agentic_retrieval")
def agentic_retrieval(req: func.HttpRequest) -> func.HttpResponse:
    # Extract 'uri' from query string
    pattern_uri = req.params.get("pattern_uri")
    if not pattern_uri:
        print(f"No pattern uri for object to be detected found.")
        pattern_uri = object_uri
        # return func.HttpResponse(
            # body='{"error": "Missing uri parameter"}',
            # status_code=400,
            # mimetype="application/json"
        # )
    content_uri = req.params.get("content_uri")
    if not content_uri:
        print(f"No content uri for scene to detect objects found.")
        content_uri = scene_uri
    # Extract or generate CorrelationId
    correlation_id = req.headers.get("x-correlation-id") or str(uuid.uuid4())
 
    import dbscan
    count = dbscan.count_multiple_matches(scene_uri, object_uri)
    # Build response payload
    response_body = {
        "Value": f"{count}",
        "CorrelationId": correlation_id
    }

    return func.HttpResponse(
        body=str(response_body),
        status_code=200,
        mimetype="application/json",
        headers={"x-correlation-id": correlation_id}
    )

