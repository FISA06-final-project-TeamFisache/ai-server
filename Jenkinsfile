pipeline {
    agent any

    triggers {
        pollSCM('H/5 * * * *')
    }

    environment {
        INSTANCE_ID = 'i-04efd59bca9034ee2'
        AWS_REGION  = 'ap-northeast-2'
    }

    stages {
        stage('Deploy') {
            steps {
                sh '''
                    COMMAND_ID=$(aws ssm send-command \
                        --region ${AWS_REGION} \
                        --instance-ids ${INSTANCE_ID} \
                        --document-name AWS-RunShellScript \
                        --parameters '{"commands":["export HOME=/root && git config --global --add safe.directory /opt/ai-server && docker network create wooriport-network 2>/dev/null || true && cd /opt/ai-server && git fetch origin && git reset --hard origin/service && docker compose up --build -d"]}' \
                        --query "Command.CommandId" \
                        --output text)

                    echo "Command ID: $COMMAND_ID"

                    for i in $(seq 1 20); do
                        sleep 10
                        STATUS=$(aws ssm get-command-invocation \
                            --region ${AWS_REGION} \
                            --command-id $COMMAND_ID \
                            --instance-id ${INSTANCE_ID} \
                            --query "Status" \
                            --output text)
                        echo "[$i] Status: $STATUS"
                        if [ "$STATUS" = "Success" ]; then exit 0; fi
                        if [ "$STATUS" = "Failed" ] || [ "$STATUS" = "Cancelled" ] || [ "$STATUS" = "TimedOut" ]; then exit 1; fi
                    done
                    echo "Timeout"
                    exit 1
                '''
            }
        }
    }

    post {
        success { echo 'AI server deployed successfully' }
        failure  { echo 'AI server deployment failed' }
    }
}
