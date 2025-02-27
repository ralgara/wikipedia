# CloudFormation tooling to create Wikipedia stats stack

Scripts to create a working Wikipedia stats stack using S3 and Lambda.

**NOTE:** Some of the commands described here will create actual cloud resources in AWS on behalf of the account referenced in your AWS CLI credentials configuration. Proceed only if you fully understand the implications of running this code.

## Requirements

* AWS SAM CLI
    * Install [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html). This will enable invoking `sam` from the shell to deploy CloudFormation templates.
* AWS CLI credentials
    * The recommended approach 


## Workflow

### Validate CloudFormation template

If you have modified the default template (`./template.yaml`), you should validate syntax locally first. Use:

`sam validate --lint`

If the tool reports valid syntax, you can move on to deploy the stack as shown below.

### Deploy CloudFormation template

To deploy, use:

`sam deploy --capabilities CAPABILITY_NAMED_IAM`

The parameter `--capabilities CAPABILITY_NAMED_IAM` is needed if you use the standard template as it contains IAM resources to set up permissions assuming the stack is operated in a protected development environment.