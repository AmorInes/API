call env\Scripts\activate
call waitress-serve --listen=127.0.0.1:80 --thread=12 main_test:app