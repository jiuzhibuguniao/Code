import os
import socket
import paramiko
import pysftp

'''
Name: remoteCtrl
Author: @Rooobins
Description: remote command and file transfer API Base on paramiko and pysftp
Date: 2019-7-14 22:27
'''


class remoteCtrl(object):
    # Description : remote command.
    def command(self, ip, passwd, cmd, port=22, user='root', timeout=60):
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            client.connect(hostname=ip, port=int(port), username=user, password=passwd, timeout=timeout,
                           allow_agent=False, look_for_keys=False)

        # 连接超时
        except socket.timeout as e:
            return 502, e

        # 密码错误
        except paramiko.ssh_exception.AuthenticationException:
            print("Password [%s] error" % passwd)
            client.close()
            return 403, "Password [%s] error" % passwd

        # 其他错误
        except Exception as e:
            print(e)
            # 系统重装后会出现hostkey验证失败问题，需要先删除known_hosts中记录,用法：若返回503则重新下发即可
            if "Host key for server" in str(e):
                os.system('sed -i "/^\[%s].*/d" ~/.ssh/known_hosts' % ip)
                client.close()
                return 503, e
            else:
                client.close()
                return 1, e

        # 执行命令之前设置为utf8环境，其中 1>&2 尤为重要，可以解决paramiko远程执行后台脚本僵死的问题
        # stdin, stdout, stderr = client.exec_command("export LANG=en_US.UTF-8;export LC_ALL=en_US.UTF-8;%s  1>&2" % cmd)
        stdin, stdout, stderr = client.exec_command(cmd)

        result_info = ""

        for line in stderr.readlines():  # 因为有了 1>&2，所以读的是stderr
            result_info += line

        # 返回状态码和打屏信息
        return stderr.channel.recv_exit_status(), result_info

    # Description : paramiko & pysftp & sftp transfer.
    def transfer(self, ip, passwd, src, dst, action='push', user='root', port=22, timeout=60):

        # 忽略hostkeys错误
        cnopts = pysftp.CnOpts()
        cnopts.hostkeys = None

        # 若src以斜杠结尾，则去掉这个斜杠，是否是目录后面会有判断逻辑
        if src[-1] == '/':
            src = src[0:-1]

        try:
            with pysftp.Connection(ip, username=user, password=passwd, port=int(port), cnopts=cnopts) as sftp:
                # 拉取文件或目录
                if action == 'pull':

                    try:
                        # 判断远程来源是目录还文件
                        if sftp.isdir(src):

                            # 判断本地目录是否存在，若不存在则创建
                            if not os.path.exists(dst):

                                try:
                                    os.makedirs(dst)

                                except Exception as e:
                                    print(e)

                                    pass

                            # 若为目录则分别取得父目录和需要操作的目录路径，进入父目录，然后执行sftp
                            parent_dir = src.rsplit('/', 1)[0]
                            opt_dir = src.rsplit('/', 1)[1]

                            sftp.chdir(parent_dir)
                            sftp.get_r(opt_dir, dst, preserve_mtime=True)

                        else:

                            # 拉取src远程文件到dst本地文件夹
                            if dst[-1] == '/':

                                # 判断本地目录是否存在，若不存在则创建
                                if not os.path.exists(dst):

                                    try:
                                        os.makedirs(dst)

                                    except Exception as e:
                                        print(e)

                                        pass

                                os.chdir(dst)
                                sftp.get(src, preserve_mtime=True)

                            # 拉取src远程文件到dst本地文件
                            else:
                                file_dir = dst.rsplit('/', 1)[0]
                                dst_file = dst.rsplit('/', 1)[1]  # 取得目标文件名称

                                # 判断本地目录是否存在，若不存在则创建
                                if not os.path.exists(file_dir):

                                    try:
                                        os.makedirs(file_dir)

                                    except Exception as e:
                                        print(e)

                                        pass

                                os.chdir(file_dir)
                                sftp.get(src, dst_file, preserve_mtime=True)

                    except Exception as e:
                        return 1, e

                else:

                    try:
                        # 判断本地文件是目录还是文件，若是目录则使用put_r 递归推送
                        if os.path.isdir(src):

                            # 判断目的目录是否存在，若不存在则创建
                            if not sftp.exists(dst):

                                try:
                                    sftp.makedirs(dst)

                                except Exception as e:
                                    print(e)

                                    pass

                            sftp.put_r(src, dst, preserve_mtime=True)

                        # 否则先进入目标目录，然后使用put单文件推送
                        else:
                            # 推送src源文件到dst目的文件夹
                            if dst[-1] == '/':

                                # 判断目的目录是否存在，若不存在则创建
                                if not sftp.exists(dst):

                                    try:
                                        sftp.makedirs(dst)

                                    except Exception as e:
                                        print(e)

                                        pass

                                sftp.chdir(dst)
                                sftp.put(src, preserve_mtime=True)

                            # 推送src源文件到dst目的文件
                            else:
                                file_dir = dst.rsplit('/', 1)[0]

                                # 判断目的目录是否存在，若不存在则创建
                                if not sftp.exists(file_dir):

                                    try:
                                        sftp.makedirs(file_dir)

                                    except Exception as e:
                                        print(e)

                                        pass

                                sftp.chdir(file_dir)
                                sftp.put(src, dst, preserve_mtime=True)

                    except Exception as e:
                        return 1, e

                return 0, 'success'

        except socket.timeout as e:
            return 502, e

        except paramiko.ssh_exception.AuthenticationException:
            print("Password [%s] error" % passwd)
            paramiko.SSHClient().close()
            # client.close()
            return 403, "Password [%s] error" % passwd

        except Exception as e:
            print(e)

            # 系统重装后会出现hostkey验证失败问题，需要先删除known_hosts中记录
            if "Host key for server" in str(e):
                os.system('sed -i "/^\[%s].*/d" ~/.ssh/known_hosts' % ip)
                paramiko.SSHClient().close()
                # client.close()
                return 503, 'Hostkeys Error'
            else:
                paramiko.SSHClient().close()
                # client.close()
                return 1, e