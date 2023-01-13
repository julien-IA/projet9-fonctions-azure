import logging
import azure.functions as func


def main(req: func.HttpRequest, outputBlob) -> func.HttpResponse:
    name = req.params.get('name')
    site_response = str(req.get_body())
    outputBlob.set(req.get_body())

    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('name')

    if name:
        return func.HttpResponse(str(site_response))
    else:
        return func.HttpResponse(
            "Please pass a name on the query string or in the request body",
            status_code=400
        )
