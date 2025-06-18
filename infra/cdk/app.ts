import { App } from 'aws-cdk-lib';
import { S3SpaStack } from './s3_spa_stack';
import { EcsStack } from './ecs_stack';

const app = new App();
new S3SpaStack(app, 'S3SpaStack');
new EcsStack(app, 'EcsStack');
