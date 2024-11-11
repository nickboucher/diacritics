SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker compose --project-directory $SCRIPT_DIR up -d --build
CONTAINER=$(docker ps -aqf "name=diacritics-previous-work-jupyter")
sleep 5
echo $(docker logs $CONTAINER 2>&1 | grep -E "^\s*http://127.0.0.1")